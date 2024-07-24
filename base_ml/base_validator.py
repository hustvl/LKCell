# -*- coding: utf-8 -*-
# Validators


from schema import Schema, Or

sweep_schema = Schema(
    {
        "method": Or("grid", "random", "bayes"),
        "name": str,
        "metric": {"name": str, "goal": Or("maximize", "minimize")},
        "run_cap": int,
    },
    ignore_extra_keys=True,
)
