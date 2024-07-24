# -*- coding: utf-8 -*-
# Base Machine Learning Experiment


import copy
import inspect
import logging
import os
import random
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Tuple, Union
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from pydantic import BaseModel
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, _LRScheduler
from torch.utils.data import Dataset, Sampler

from base_ml.base_optim import OPTI_DICT
from base_ml.base_validator import sweep_schema
from utils.logger import Logger
from utils.tools import flatten_dict, remove_parameter_tag, unflatten_dict

from base_ml.optim_factory import LayerDecayValueAssigner, create_optimizer


class BaseExperiment:
    """BaseExperiment Class

    An experiment consistsn of the follwing key methods:

        * run_experiment: Main Code for running the experiment with implemented coordinaten and training call
        *
        *
    Args:
        default_conf (dict): Default configuration
    """

    def __init__(self, default_conf: dict, checkpoint=None) -> None:
        # setup configuration
        self.default_conf = default_conf
        self.run_conf = None
        self.logger = logging.getLogger(__name__)

        # resolve_paths
        self.default_conf["logging"]["log_dir"] = str(
            Path(default_conf["logging"]["log_dir"]).resolve()
        )
        self.default_conf["logging"]["wandb_dir"] = str(
            Path(default_conf["logging"]["wandb_dir"]).resolve()
        )

        if checkpoint is not None:
            self.checkpoint = torch.load(checkpoint, map_location="cpu")
        else:
            self.checkpoint = None

        # seeding
        self.seed_run(seed=self.default_conf["random_seed"])

    @abstractmethod
    def run_experiment(self):
        """Experiment Code

        Main Code for running the experiment. The following steps should be performed:
        1.) Set run name
        2.) Initialize WandB and update config (According to Sweep or predefined)
        3.) Create Output directory and setup logger
        4.) Machine Learning Setup
            4.1) Loss functions
            4.2) Model
            4.3) Optimizer
            4.4) Scheduler
        5.) Load and Setup Dataset
        6.) Define Trainer
        7.) trainer.fit()

        Raises:
            NotImplementedError: Needs to be implemented
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_model(self) -> nn.Module:
        """Retrieve torch model for training

        Raises:
            NotImplementedError: Needs to be implemented

        Returns:
            nn.Module: Torch Model
        """
        raise NotImplementedError

    @abstractmethod
    def get_loss_fn(self) -> _Loss:
        """Retrieve torch loss function for training

        Raises:
            NotImplementedError: Needs to be implemented

        Returns:
            _Loss: Loss function
        """
        raise NotImplementedError
    
    def get_argparser():
        parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)

    # Optimization parameters
        parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                            help='Optimizer (default: "adamw"')
        parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                            help='Optimizer Epsilon (default: 1e-8)')
        parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                            help='Optimizer Betas (default: None, use opt default)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=0.05,
                            help='weight decay (default: 0.05)')
        parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
            weight decay. We use a cosine schedule for WD and using a larger decay by
            the end of training improves performance for ViTs.""")    

        parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                            help='learning rate (default: 4e-3), with total batch size 4096')
        parser.add_argument('--layer_decay', type=float, default=0.9999)


        return parser




    def get_optimizer(
        self, model: nn.Module, opt: str, hp: dict, layer_decay:float, 
    ) -> Optimizer:
        """Retrieve optimizer for training

        All Torch Optimizers are possible

        Args:
            model (nn.Module): Training model
            optimizer_name (str): Name of the optimizer, all current PyTorch Optimizer are possible
            hp (dict): Hyperparameter as dictionary. For further information,
                see documentation here: https://pytorch.org/docs/stable/optim.html#algorithms

        Raises:
            NotImplementedError: Raises error if an undefined Optimizer differing from torch is used

        Returns:
            Optimizer: PyTorch Optimizer
        """
        # if optimizer_name not in OPTI_DICT:
        #     raise NotImplementedError("Optimizer not known")

        if layer_decay < 1.0 or layer_decay > 1.0:
            num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
            
            assigner = LayerDecayValueAssigner(list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        #optim = OPTI_DICT[optimizer_name]
        # optimizer = optim(
        #     params=filter(lambda p: p.requires_grad, model.parameters()), **hp
        # )
        #optimizer = optim(params=model.parameters(), **hp)
        
        optimizer = create_optimizer(
            model, weight_decay=hp["weight_decay"], lr=hp["lr"], opt=opt, get_num_layer=assigner.get_layer_id, get_layer_scale=assigner.get_scale)
        
        self.logger.info(
            f"Loaded  Optimizer with following hyperparameters:"
        )
        self.logger.info(hp)

        return optimizer

    def get_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        """Retrieve learning rate scheduler for training

        Currently, just constant scheduler. Should be extended to add a configurable scheduler.
        Maybe reimplement in specific experiment file.

        Args:
            optimizer (Optimizer): Optimizer

        Returns:
            _LRScheduler: PyTorch Scheduler
        """
        scheduler = ConstantLR(optimizer, factor=1, total_iters=1000)
        self.logger.info("Scheduler: ConstantLR scheduler")
        return scheduler

    def get_sampler(self) -> Sampler:
        """Retrieve data sampler for training

        Raises:
            NotImplementedError: Needs to be implemented

        Returns:
            Sampler: Training sampler
        """
        raise NotImplementedError

    def get_train_dataset(self) -> Dataset:
        """Retrieve training dataset

        Raises:
            NotImplementedError: Needs to be implemented

        Returns:
            Dataset: Training dataset
        """
        raise NotImplementedError

    def get_val_dataset(self) -> Dataset:
        """Retrieve validation dataset

        Raises:
            NotImplementedError: Needs to be implemented

        Returns:
            Dataset: Validation dataset
        """
        raise NotImplementedError

    def load_file_split(
        self, fold: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the file split for training, validation and test

        If no fold is provided, the current file split is loaded. Otherwise the files in the fold are loaded

        The folder (filelist_path) must be built up in the following way:
            1.) No-Multifold:
            filelist_path:
                train_split.csv
                val_split.csv
                test_split.csv
            2.) Multifold:
            filelist_path:
                fold1:
                    train_split.csv
                    val_split.csv
                    test_split.csv
                fold2:
                    train_split.csv
                    val_split.csv
                    test_split.csv
                ...
                foldN:
                    train_split.csv
                    val_split.csv
                    test_split.csv

        Args:
            fold (int, optional): Fold. Defaults to None.

        Raises:
            NotImplementedError: Fold selection is currently not Implemented

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, Val and Test split as Pandas Dataframe
        """
        filelist_path = Path(self.default_conf["split_path"]).resolve()
        self.logger.info(f"Loading filesplit from folder: {filelist_path}")
        if fold is None:
            train_split = pd.read_csv(filelist_path / "train_split.csv")
            val_split = pd.read_csv(filelist_path / "val_split.csv")
            test_split = pd.read_csv(filelist_path / "test_split.csv")
        else:
            train_split = pd.read_csv(filelist_path / f"fold{fold}" / "train_split.csv")
            val_split = pd.read_csv(filelist_path / f"fold{fold}" / "val_split.csv")
            test_split = None

        self.logger.info(f"Train size: {len(train_split)}")
        self.logger.info(f"Val-Split: {len(val_split)}")
        return train_split, val_split, test_split

    # Methods regarding logging and storing
    def instantiate_logger(self) -> Logger:
        """Instantiate a logger

        Returns:
            Logger: Logger
        """
        logger = Logger(
            level=self.default_conf["logging"]["level"].upper(),
            log_dir=Path(self.run_conf["logging"]["log_dir"]).resolve(),
            comment="logs",
            use_timestamp=False,
        )
        self.logger = logger.create_logger()
        return self.logger

    @staticmethod
    def create_output_dir(folder_path: Union[str, Path]) -> None:
        """Create folder at given path

        Args:
            folder_path (Union[str, Path]): Folder that should be created
        """
        folder_path = Path(folder_path).resolve()
        folder_path.mkdir(parents=True, exist_ok=True)

    def store_config(self) -> None:
        """Store the config file in the logging directory to keep track of the configuration."""
        # store in log directory
        with open(
            (Path(self.run_conf["logging"]["log_dir"]) / "config.yaml").resolve(), "w"
        ) as yaml_file:
            tmp_config = copy.deepcopy(self.run_conf)
            tmp_config["logging"]["log_dir"] = str(tmp_config["logging"]["log_dir"])

            yaml.dump(tmp_config, yaml_file, sort_keys=False)

        self.logger.debug(
            f"Stored config under: {(Path(self.run_conf['logging']['log_dir']) / 'config.yaml').resolve()}"
        )

    @staticmethod
    def extract_sweep_arguments(config: dict) -> Tuple[Union[BaseModel, dict]]:
        """Extract sweep argument from the provided dictionary

        The config dictionary must contain a "sweep" entry with the sweep configuration.
        The file structure is documented here: ./base_ml/base_validator.py
        We follow the official sweep guidlines of WandB
        Example Sweep files are provided in the ./configs/examples folder

        Args:
            config (dict): Dictionary with all configurations

        Raises:
            KeyError: Missing Sweep Keys

        Returns:
            Tuple[Union[BaseModel, dict]]: Sweep arguments
        """
        # validate sweep settings
        if "sweep" not in config:
            raise KeyError("No Sweep configuration provided")
        sweep_schema.validate(config["sweep"])

        sweep_conf = config["sweep"]

        # load parameters
        flattened_dict = flatten_dict(config, sep=".")
        filtered_dict = {
            k: v for k, v in flattened_dict.items() if "parameters" in k.split(".")
        }
        parameters = remove_parameter_tag(filtered_dict, sep=".")

        sweep_conf["parameters"] = parameters

        return sweep_conf

    def overwrite_sweep_values(self, run_conf: dict, sweep_run_conf: dict) -> None:
        """Overwrite run_conf file with the sweep values

        For the sweep, sweeping parameters are a flattened dict, with keys beeing specific with '.' separator.
        These dictionary with the sweep hyperparameter selection needs to be unflattened (convert '.' into nested dict)
        Afterward, keys are insertd in the run_conf dictionary

        Args:
            run_conf (dict): Current dictionary without sweep selected parameters
            sweep_run_conf (dict): Dictionary with the sweep config
        """
        flattened_run_conf = flatten_dict(run_conf, sep=".")
        filtered_dict = {
            k: v
            for k, v in flattened_run_conf.items()
            if "parameters" not in k.split(".")
        }
        run_parameters = {**filtered_dict, **sweep_run_conf}
        run_parameters = unflatten_dict(run_parameters, ".")

        self.run_conf = run_parameters

    @staticmethod
    def seed_run(seed: int) -> None:
        """Seed the experiment

        Args:
            seed (int): Seed
        """
        # seeding
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        from packaging.version import parse, Version

        try:
            import tensorflow as tf
        except ImportError:
            pass
        else:
            if parse(tf.__version__) >= Version("2.0.0"):
                tf.random.set_seed(seed)
            elif parse(tf.__version__) <= Version("1.13.2"):
                tf.set_random_seed(seed)
            else:
                tf.compat.v1.set_random_seed(seed)

    @staticmethod
    def seed_worker(worker_id) -> None:
        """Seed a worker

        Args:
            worker_id (_type_): Worker ID
        """
        worker_seed = torch.initial_seed() % 2**32
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def close_remaining_logger(self) -> None:
        """Close all remaining loggers"""
        logger = logging.getLogger("__main__")
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.handlers.clear()
        logging.shutdown()
