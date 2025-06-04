"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--is_tta", default=False, help="whether or not to use TTA")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    # runner.evaluate(skip_reload=True)
    runner.evaluate_tta(skip_reload=True, tta_cfg=cfg.config.tta)


if __name__ == "__main__":
    main()


# TTA itm_adapt
# - debug args: --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml --is_tta True
# - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t.out 2>&1 &
# - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out 2>&1 &