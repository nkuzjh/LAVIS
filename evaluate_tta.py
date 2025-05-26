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

# TTA tent:
# debug args: --cfg-path lavis/projects/blip2/eval/ret_coco_eval_tent.yaml --is_tta True
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node=1 ../evaluate_tta.py --tta True --cfg-path ../lavis/projects/blip2/eval/ret_coco_eval_tent.yaml


# TTA zhh:
# debug args: --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh.yaml --is_tta True
# CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh.yaml >ret_coco_eval_zhh.out &

# {'txt_r1': 77.84, 'txt_r5': 94.2, 'txt_r10': 97.1, 'txt_r_mean': 89.71333333333332, 'img_r1': 58.220711715313875, 'img_r5': 81.17153138744503, 'img_r10': 87.68092762894842, 'img_r_mean': 75.6910569105691, 'r_mean': 82.7021951219512, 'agg_metrics': 89.71333333333332}

# cosine similarity结果除以temperature，再作为uncertainty
# tent lr=1e-4 wd=0.0
# {'txt_r1': 82.18, 'txt_r5': 95.7, 'txt_r10': 97.9, 'txt_r_mean': 91.92666666666666, 'img_r1': 65.33386645341864, 'img_r5': 86.29348260695721, 'img_r10': 91.84326269492203, 'img_r_mean': 81.1568705850993, 'r_mean': 86.54176862588298, 'agg_metrics': 91.92666666666666}

# cosine similarity结果除以temperature，再作为uncertainty
# tent lr=1e-4 wd=0.0
# tta loss中去除 1-uncertainty 这一项
# {"txt_r1": 82.18, "txt_r5": 95.7, "txt_r10": 97.9, "txt_r_mean": 91.92666666666666, "img_r1": 65.33386645341864, "img_r5": 86.29348260695721, "img_r10": 91.84326269492203, "img_r_mean": 81.1568705850993, "r_mean": 86.54176862588298, "agg_metrics": 91.92666666666666}
# 结果无变化，因为visual_encoder和text_encoder在计算cos_sim时是torch.no_grad()的，且encoder的model.weight不更新。

# cosine similarity结果除以temperature，再作为uncertainty
# tent lr=1e-4 wd=0.0
# Qformer + Tent without Uncertainty
# {"txt_r1": 82.36, "txt_r5": 95.74, "txt_r10": 97.96, "txt_r_mean": 92.02, "img_r1": 66.01759296281487, "img_r5": 86.74130347860856, "img_r10": 91.96321471411436, "img_r_mean": 81.57403705184593, "r_mean": 86.79701852592297, "agg_metrics": 92.02}


# TTA zhh topk:
# debug args: --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh_topk.yaml --is_tta True
# CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh_topk.yaml >ret_coco_eval_zhh_topk.out &

# {"txt_r1": 82.34, "txt_r5": 95.76, "txt_r10": 98.02, "txt_r_mean": 92.04, "img_r1": 65.98960415833666, "img_r5": 86.61335465813674, "img_r10": 91.99120351859256, "img_r_mean": 81.53138744502199, "r_mean": 86.785693722511, "agg_metrics": 92.04}

# 增加itm tta loss的梯度累积，accumulate batch size = 64
