# -*- coding: utf-8 -*-
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from json import decoder
from pathlib import Path
from typing import List, Literal, Tuple, Union, Optional, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

from monai.networks.blocks import UpSample
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_act_layer
from monai.networks.nets.basic_unet import UpCat,TwoConv
from monai.utils import InterpolateMode
from .cellvit_unirepLKnet import UniRepLKNet
from .replknet import  *

from huggingface_hub import PyTorchModelHubMixin


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x




encoder_feature_channel = {
    "unireplknet_n": (80, 160, 320, 640),
    "unireplknet_s": (96, 192, 384, 768),
    "resnet50": (64, 256, 512, 1024, 2048),
    "vitdet_small": (768, 768, 768, 768, 768),
}



def _get_encoder_channels_by_backbone(backbone: str, in_channels: int = 3) -> tuple:
    """
    Get the encoder output channels by given backbone name.

    Args:
        backbone: name of backbone to generate features, can be from [efficientnet-b0, ..., efficientnet-b7].
        in_channels: channel of input tensor, default to 3.

    Returns:
        A tuple of output feature map channels' length .
    """
    encoder_channel_tuple = encoder_feature_channel[backbone]  #encoder_channel_tuple是指的编码器通道元组,在这里是有4个通道的元组
    encoder_channel_list = [in_channels] + list(encoder_channel_tuple)  #encoder_channel_list是指的编码器通道列表[3,80,160,320,640]
    encoder_channel = tuple(encoder_channel_list) #encoder_channel是指的编码器通道元组(3,80,160,320,640)
    return encoder_channel



class RepLKDeocder(nn.Module):
    def __init__(self,  
        encoder_channels: Sequence[int],
        spatial_dims: int, 
        decoder_channels: Sequence[int], 
        stage_lk_sizes, 
        drop_path,
        upsample: str,
        pre_conv: Optional[str],
        interp_mode: str,
        align_corners: Optional[bool],
        small_kernel,
        dw_ratio: int=1, 
        small_kernel_merged=False,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        dropout: Union[float, tuple] = 0.0,
        bias: bool = False,
        is_pad: bool = True,  
        ffn_ratio=4,  
        ):
        super().__init__()

        in_channels = [encoder_channels[-1]] + list(decoder_channels[:-1])  #in_channels=[640,1024,512,256,128]
        skip_channels = list(encoder_channels[1:-1][::-1]) + [0]
        halves = [True] * (len(skip_channels) - 1)
        halves.append(False)
        stage_lk_sizes = stage_lk_sizes
        blocks = []
        for in_chn, skip_chn, out_chn, halve in zip(in_channels, skip_channels, decoder_channels, halves):
            blocks.append(
                UpCat(
                    spatial_dims=spatial_dims,
                    in_chns=in_chn,
                    cat_chns=skip_chn,
                    out_chns=out_chn,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    upsample=upsample,
                    pre_conv=pre_conv,
                    interp_mode=interp_mode,
                    align_corners=align_corners,
                    halves=halve,
                    is_pad=is_pad,
                )
                
            )   
           
        self.blocks = nn.ModuleList(blocks)  
        repblock = []
        for i in range(4):
            repblock.append(RepLKBlock(in_channels=in_channels[i], dw_channels=int(in_channels[i] * dw_ratio), block_lk_size=stage_lk_sizes[i],
                                 small_kernel=small_kernel, drop_path=drop_path, small_kernel_merged=small_kernel_merged))
            
        self.repblock = nn.ModuleList(repblock)

        convffnblock = []
        for i in range(4):
            convffnblock.append(ConvFFN(in_channels=in_channels[i], internal_channels=int(in_channels[i] * ffn_ratio), out_channels=in_channels[i], drop_path=drop_path))
        self.convffnblock = nn.ModuleList(convffnblock)
        
        self.upsample = [UpSample(
            spatial_dims, 
            decoder_channels[i],
            decoder_channels[i],
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,) for i in range(len(decoder_channels) - 1)]  
        
        self.upsample1= UpSample(
            spatial_dims,
            256,
            256,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )       
        self.upsample2= UpSample(
            spatial_dims,
            128,
            128,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )         
        self.convs = TwoConv(spatial_dims, 304, decoder_channels[-2], act, norm, bias, dropout)
        self.convs1 = TwoConv(spatial_dims, 152, decoder_channels[-1], act, norm,bias, dropout)

    def forward(self, features: List[torch.Tensor], input_feature: torch.Tensor, skip_connect: int = 3):
        skips = features[:-1][::-1]  #skips[0],[1],[2]=[16,320,16,16],[16,160,32,32],[16,80,64,64],[16,40,128,128]
        features = features[1:][::-1] 
        #input_feature = self.conv1(input_feature)  #input_feature=[16,64,256,256]
        x = features[0]  #x=[16,640,8,8]
        for i, (block, repblock, convffnblock) in enumerate(zip(self.blocks, self.repblock, self.convffnblock)):

            if i < skip_connect:
                skip = skips[i]  #skip=[16,320,16,16], skip=[16,160,32,32], skip=[16,80,64,64], skip = [16,40,128,128]
                #x = repblock(x)
                #x = convffnblock(x)
                x = block(x, skip)
                        
            else:
                #x = repblock(x)     
                skip = input_feature[1] 
                x =self.upsample1(x)
                x = torch.cat([skip, x], dim=1)
                x = self.convs(x) 
                
                skip = input_feature[0]
                x = self.upsample2(x)
                x = torch.cat([skip, x], dim=1)
                x = self.convs1(x)
 
        return x
     


class SegmentationHead(nn.Sequential):
    """
    Segmentation head.
    This class refers to `segmentation_models.pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels for the block.
        out_channels: number of output channels for the block.
        kernel_size: kernel size for the conv layer.
        act: activation type and arguments.
        scale_factor: multiplier for spatial size. Has to match input size if it is a tuple.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        act: Optional[Union[Tuple, str]] = None,
        scale_factor: float = 1.0,
    ):
        
        conv_layer = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        bn_layer = nn.BatchNorm2d(in_channels)
        conv_layer1 = conv_bn(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, groups=1)
        nonlinear_layer = nn.GELU()
        conv_layer2 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)



        up_layer: nn.Module = nn.Identity()
        if scale_factor > 1.0:
            up_layer = UpSample(
                spatial_dims=spatial_dims,
                scale_factor=scale_factor,
                mode="nontrainable",
                pre_conv=None,
                interp_mode=InterpolateMode.LINEAR,
            )
        if act is not None:
            act_layer = get_act_layer(act)
        else:
            act_layer = nn.Identity()
        super().__init__(bn_layer, conv_layer1, nonlinear_layer, conv_layer2, up_layer)





class CellViT(UniRepLKNet,PyTorchModelHubMixin):
    
    def __init__(
        self,
        model256_path: Union[Path, str],
        num_nuclei_classes=int,
        num_tissue_classes=int,
        in_channels=int,
        backbone = "unireplknet_s",
        decoder_channels: Tuple = (1024, 512, 256, 128, 64),
        spatial_dims: int = 2,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        dropout: Union[float, tuple] = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        interp_mode: str = "nearest",
        drop_path_rate: float = 0.1,
        large_kernel_sizes=[13,27,29,31],
        small_kernel=5,
        

    ):
        super().__init__()

        self.num_tissue_classes = num_tissue_classes  
        self.num_nuclei_classes = num_nuclei_classes 

        if backbone not in encoder_feature_channel:
            raise ValueError(f"invalid model_name {backbone} found, must be one of {encoder_feature_channel.keys()}.")

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims can only be 2 or 3.")


        self.backbone = backbone
        self.spatial_dims = spatial_dims
        encoder_channels = _get_encoder_channels_by_backbone(backbone, in_channels)
        self.encoder = UniRepLKNet(
            in_chans=3,
            num_classes=None,
            depths=(3, 3, 27, 3),
            dims=(96,192,384,768),
            drop_path_rate=0.3,
            layer_scale_init_value=1e-6,
            head_init_scale=1.,
            kernel_sizes=None,
            deploy=False,
            with_cp=False,
            init_cfg=None,
            attempt_use_lk_impl=True,
            use_sync_bn=False,
        )

        self.decoder = RepLKDeocder(
            encoder_channels=encoder_channels,
            stage_lk_sizes=large_kernel_sizes,
            small_kernel=small_kernel,
            drop_path=drop_path_rate,
            spatial_dims=spatial_dims,
            decoder_channels=decoder_channels,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=decoder_bias,
            upsample=upsample,
            interp_mode=interp_mode,
            pre_conv=None,
            align_corners=None,           
        ) 



        self.branches_output = {
            "nuclei_binary_map": 2,
            "hv_map": 2,
            "nuclei_type_maps": self.num_nuclei_classes,
        }  # number of output channels for each branch  

        self.nuclei_binary_segmentation_head = SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=decoder_channels[-1],
            out_channels=self.branches_output["nuclei_binary_map"],
            kernel_size=1,
            act=None,
            scale_factor=1.0,
        )

        self.hv_map_head = SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=decoder_channels[-1],
            out_channels=self.branches_output["hv_map"],
            kernel_size=1,
            act=None,
            scale_factor=1.0,
        )
        
        self.nuclei_type_maps_head = SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=decoder_channels[-1],
            out_channels=self.branches_output["nuclei_type_maps"],
            kernel_size=1,
            act=None,
            scale_factor=1.0,
        )



        self.model256_path = model256_path

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass of CellViT  

        Args:
            x (torch.Tensor): Images in BCHW style 
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well. Defaults to False.   

        Returns: 
            dict: Output for all branches:  
                * tissue_types: Raw tissue type prediction. Shape: (batch_size, num_tissue_classes) 
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (batch_size, 2, H, W)  
                * hv_map: Binary HV Map predictions. Shape: (batch_size, 2, H, W) 
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (batch_size, num_nuclei_classes, H, W)  
                * (optinal) tokens 
        """
        

        out_dict = {}  
        classifier_logits, z, input_feature = self.encoder(x)
        
        out_dict["tissue_types"] = classifier_logits   
        decoder_output = self.decoder(z,input_feature)
        
     

        out_dict["nuclei_binary_map"] = self.nuclei_binary_segmentation_head(decoder_output)
        out_dict["hv_map"] = self.hv_map_head(decoder_output)
        out_dict["nuclei_type_map"] = self.nuclei_type_maps_head(decoder_output)
        

        return out_dict
    


    def load_pretrained_encoder(self, model256_path: str):
        """Load pretrained ViT-256 from provided path

        Args:
            model256_path (str): Path to ViT-256
        """
        state_dict = torch.load(str(model256_path), map_location="cpu")
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith("head")}
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")




    def reshape_model_output(
        self,
        predictions: OrderedDict,
        device: str,
    ) -> OrderedDict:
       
        predictions = OrderedDict(
            [
                [k, v.permute(0, 2, 3, 1).contiguous().to(device)]  
                for k, v in predictions.items()  
                if k != "tissue_types" 
            ]
        )
        return predictions

    def calculate_instance_map(
        self, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Calculate Instance Map from network predictions (after Softmax output) 

        Args:
            predictions (dict): Dictionary with the following required keys:
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (batch_size, H, W, 2)  
                * nuclei_type_map: Type prediction of nuclei. Shape: (batch_size, H, W, 6) 
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (batch_size, H, W, 2)  
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.  

        Returns:
            Tuple[torch.Tensor, List[dict]]:  
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (batch_size, H, W) 
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus. 
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"  
        """
         # reshape to B, H, W, C
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)
        predictions = predictions_
        
        cell_post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes, magnification=magnification, gt=False 
        ) 
        instance_preds = []
        type_preds = []
        for i in range(predictions["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions["nuclei_type_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    torch.argmax(predictions["nuclei_binary_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    predictions["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map) 
            instance_preds.append(instance_pred[0])  
            type_preds.append(instance_pred[1])  

        return torch.Tensor(np.stack(instance_preds)), type_preds 
    
    

    def generate_instance_nuclei_map(
        self, instance_maps: torch.Tensor, type_preds: List[dict]
    ) -> torch.Tensor:
        """Convert instance map (binary) to nuclei type instance map  
        Args:
            instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (batch_size, H, W) 
            type_preds (List[dict]): List (len=batch_size) of dictionary with instance type information (compare post_process_hovernet function for more details)  

        Returns:
            torch.Tensor: Nuclei type instance map. Shape: (batch_size, H, W, self.num_nuclei_classes)   
        """
        batch_size, h, w = instance_maps.shape 
        instance_type_nuclei_maps = torch.zeros(
            (batch_size, h, w, self.num_nuclei_classes)
        )  
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][
                    instance_map == nuclei
                ] = nuclei   

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map  
        return instance_type_nuclei_maps

    def freeze_encoder(self):
        """Freeze encoder to not train it """  
        for layer_name, p in self.encoder.named_parameters(): 
            if layer_name.split(".")[0] != "head":  
                p.requires_grad = False 

    def unfreeze_encoder(self):
        """Unfreeze encoder to train the whole model """
        for p in self.encoder.parameters():
            p.requires_grad = True



@dataclass
class DataclassHVStorage:
    """Storing PanNuke Prediction/GT objects for calculating loss, metrics etc. with HoverNet networks

    Args:
        nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
        hv_map (torch.Tensor): Logit output for HV-Map. Shape: (batch_size, 2, H, W)
        nuclei_type_map (torch.Tensor): Softmax output for nuclei type-prediction. Shape: (batch_size, num_tissue_classes, H, W)
        tissue_types (torch.Tensor): Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
        instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
            Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
        instance_types_nuclei: Pixel-wise nuclear instance segmentation predictions, for each nuclei type.
            Each instance has its own integer, starting from 1.
            Shape: (batch_size, num_nuclei_classes, H, W)
        batch_size (int): Batch size of the experiment
        instance_types (list, optional): Instance type prediction list.
            Each list entry stands for one image. Each list entry is a dictionary with the following structure:
            Main Key is the nuclei instance number (int), with a dict as value.
            For each instance, the dictionary contains the keys: bbox (bounding box), centroid (centroid coordinates),
            contour, type_prob (probability), type (nuclei type)
            Defaults to None.
        regression_map (torch.Tensor, optional): Regression map for binary prediction map.
            Shape: (batch_size, 2, H, W). Defaults to None.
        regression_loss (bool, optional): Indicating if regression map is present. Defaults to False.
        h (int, optional): Height of used input images. Defaults to 256.
        w (int, optional): Width of used input images. Defaults to 256.
        num_tissue_classes (int, optional): Number of tissue classes in the data. Defaults to 19.
        num_nuclei_classes (int, optional): Number of nuclei types in the data (including background). Defaults to 6.
    """

    nuclei_binary_map: torch.Tensor
    hv_map: torch.Tensor
    tissue_types: torch.Tensor
    nuclei_type_map: torch.Tensor
    instance_map: torch.Tensor
    instance_types_nuclei: torch.Tensor
    batch_size: int
    instance_types: list = None
    regression_map: torch.Tensor = None
    regression_loss: bool = False
    h: int = 256
    w: int = 256
    num_tissue_classes: int = 19
    num_nuclei_classes: int = 6



    def get_dict(self) -> dict:
        """Return dictionary of entries"""
        property_dict = self.__dict__
        if not self.regression_loss and "regression_map" in property_dict.keys():
            property_dict.pop("regression_map")
        return property_dict
