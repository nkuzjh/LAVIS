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
from lavis.runners.runner_tta import RunnerTTA
from lavis.tasks import *


from tta.datasets import create_tta_dataset, create_eval_dataset



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
    seed = config.run_cfg.seed #+ get_rank()

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

    
    from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
    model = Blip2Qformer() # .from_config(cfg=cfg.model)
    model = task.build_model(cfg)
    
    eval_dataset = create_eval_dataset(cfg)
    eval_dataloader = create_eval_dataloader(eval_dataset)
    sim_matrix_i2t, sim_matrix_t2i, image_embeds, vit_feats, text_embeds, text_ids, text_atts = compute_embeds(model, eval_dataloader, task_cfg, tta_cfg)
    # np.save("debugs/debug_sim_matrix_i2t.npy",sim_matrix_i2t.numpy())
    # np.save("debugs/debug_sim_matrix_t2i.npy",sim_matrix_t2i.numpy())
    # np.save("debugs/debug_image_embeds.npy",image_embeds.numpy())
    # np.save("/data/jiahao/blip2_embeddings/debug_vit_feats.npy",vit_feats.numpy())
    # np.save("debugs/debug_text_embeds.npy",text_embeds.numpy())
    # np.save("debugs/debug_text_ids.npy",text_ids.numpy())
    # np.save("debugs/debug_text_atts.npy",text_atts.numpy())

    tta_dataset = create_tta_dataset(cfg)



    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)#{'train': <lavis.datasets.datasets.base_dataset.ConcatDataset object at 0x7f55a0b90b90>, 'val': <lavis.datasets.datasets.retrieval_datasets.RetrievalEvalDataset object at 0x7f5548832d90>, 'test': <lavis.datasets.datasets.retrieval_datasets.RetrievalEvalDataset object at 0x7f5547cddbd0>}
    

    # runner = RunnerBase(
    #     cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets,
    # )
    runner = RunnerTTA(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets,
    )
    # runner.evaluate(skip_reload=True)
    runner.evaluate_tta(tta_cfg=cfg.config.tta)


if __name__ == "__main__":
    main()



# TTA itm_adapt
# --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.yaml --is_tta True
# --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i_exp11.yaml 
# - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t.out 2>&1 &
# - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out 2>&1 &



###########################
###########################
###########################

# TTA tent:
# debug args: --cfg-path lavis/projects/blip2/eval/ret_coco_eval_tent.yaml --is_tta True
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node=1 ../evaluate_tta.py --tta True --cfg-path ../lavis/projects/blip2/eval/ret_coco_eval_tent.yaml

# TTA zhh:
# debug args: --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh.yaml --is_tta True
# CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh.yaml >ret_coco_eval_zhh_wotemper.out &

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

# cosine similarity结果除以temperature，再作为uncertainty
# tent lr=1e-4 wd=0.0
# Qformer + Tent without Uncertainty
# gradient accumulation steps = 64 ??? 后续发现代码没有实现 grad_accu
# {'txt_r1': 82.36, 'txt_r5': 95.74, 'txt_r10': 97.96, 'txt_r_mean': 92.02, 'img_r1': 66.01759296281487, 'img_r5': 86.74130347860856, 'img_r10': 91.96321471411436, 'img_r_mean': 81.57403705184593, 'r_mean': 86.79701852592297, 'agg_metrics': 92.02}

# cosine similarity结果除以temperature，再作为uncertainty
# tent lr=1e-4 wd=0.0
# Qformer + Tent without Uncertainty
# rerank_score = itm_score +　cosine similarity without dividing temperature
# is same to itm_adapt without top1_match_coeffi but with grad_acc, run at remote 58


# cosine similarity结果除以temperature，再作为uncertainty
# tent lr=5e-6 wd=0.0
# Qformer + Tent without Uncertainty
# rerank_score = itm_score +　cosine similarity without dividing temperature
# is same to itm_adapt without top1_match_coeffi but with grad_acc, run at remote 58



# TTA zhh_topk
# 互相top1-topk的概率均值作为unc加权:
# debug args: --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh_topk.yaml --is_tta True
# CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh_topk.yaml >ret_coco_eval_zhh_topk_wotemper_i2t.out &
# CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh_topk_t2i.yaml >ret_coco_eval_zhh_topk_wotemper_t2i.out &

# {"txt_r1": 82.34, "txt_r5": 95.76, "txt_r10": 98.02, "txt_r_mean": 92.04, "img_r1": 65.98960415833666, "img_r5": 86.61335465813674, "img_r10": 91.99120351859256, "img_r_mean": 81.53138744502199, "r_mean": 86.785693722511, "agg_metrics": 92.04}

# 增加itm tta loss的梯度累积，accumulate batch size = 64
# {"txt_r1": 83.02, "txt_r5": 96.34, "txt_r10": 98.2, "txt_r_mean": 92.52, "img_r1": 67.46501399440224, "img_r5": 87.51299480207916, "img_r10": 92.67093162734906, "img_r_mean": 82.54964680794349, "r_mean": 87.53482340397174, "agg_metrics": 92.52}

# 增加itm tta loss的梯度累积，accumulate batch size = 64
# offline, online=False
# {'txt_r1': 84.18, 'txt_r5': 96.22, 'txt_r10': 98.26, 'txt_r_mean': 92.88666666666667, 'img_r1': 66.57736905237905, 'img_r5': 86.76929228308677, 'img_r10': 92.04318272690924, 'img_r_mean': 81.79661468745836, 'r_mean': 87.34164067706251, 'agg_metrics': 92.88666666666667}

# lr=5e-6 wd=0.0
# 增加itm tta loss的梯度累积，accumulate batch size = 64
# offline, online=False
# multi_epochs=5
# epoch 0: {"txt_r1": 84.06, "txt_r5": 96.42, "txt_r10": 98.1, "txt_r_mean": 92.86000000000001, "img_r1": 67.58096761295482, "img_r5": 87.3970411835266, "img_r10": 92.47101159536186, "img_r_mean": 82.48300679728109, "r_mean": 87.67150339864055, "agg_metrics": 92.86000000000001}
# epoch 1: {"txt_r1": 84.58, "txt_r5": 96.78, "txt_r10": 98.34, "txt_r_mean": 93.23333333333335, "img_r1": 67.30907636945221, "img_r5": 87.19312275089965, "img_r10": 92.39504198320672, "img_r_mean": 82.29908036785287, "r_mean": 87.76620685059311, "agg_metrics": 93.23333333333335}

# lr=5e-6 wd=0.0
# 增加itm tta loss的梯度累积，accumulate batch size = 64
# offline, online=False
# rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature):
# BUT continue ckpt from "Retrieval_COCO_zhh_lr5e-6_wd0_QfomerTentwoUnc_topk_gradacc_offline_multiepochs5/20250529105/i2t_tta_model_model_epoch_1.pth";
# report i2t metrics online, at epoch 0, : {'txt_r1': 84.68, 'txt_r5': 96.5, 'txt_r10': 98.28, 'txt_r_mean': 93.15333333333335, 'img_r1': -999, 'img_r5': -999, 'img_r10': -999, 'img_r_mean': -999, 'r_mean': -999, 'agg_metrics': -999}
# report i2t metrics offline, at epoch 0 : {'txt_r1': 84.16, 'txt_r5': 96.42, 'txt_r10': 98.2, 'txt_r_mean': 92.92666666666666, 'img_r1': 67.47700919632148, 'img_r5': 87.44102359056377, 'img_r10': 92.40703718512594, 'img_r_mean': 82.4416899906704, 'r_mean': 87.68417832866854, 'agg_metrics': 92.92666666666666}

# lr=5e-6 wd=0.0
# 增加itm tta loss的梯度累积，accumulate batch size = 64
# offline, online=False
# rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
# multi_epochs=5
# i2t best:
    # report i2t metrics online, at epoch 2 :
    # "txt_r1": 85.08, "txt_r5": 96.62, "txt_r10": 98.32, "txt_r_mean": 93.33999999999999, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
# i2t best:
    # report t2i metrics online, at epoch 2 :s
    # {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.4970011995202, "img_r5": 87.3890443822471, "img_r10": 92.5389844062375, "img_r_mean": 82.47500999600159, "r_mean": -999, "agg_metrics": -999}





# TTA zhh_topk_ss
# 互相top1-topk的概率均值作为unc加权
# 增加互相recall时的sample selection
# debug args: --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh_topk_ss.yaml --is_tta True
# CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_zhh_topk_ss.yaml >ret_coco_eval_zhh_topk_ss.out &
# implemented by itm_adapt_ss
