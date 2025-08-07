from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
import logging
import numpy as np
import os
import json

from lavis.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from .utils import (
    compute_embeds,
    compute_kl_coeffis,
    compute_i2t_itm_score,
    adapt_i2t_itm_score,
    adapt_i2t_itm_score_v2,
    compute_i2t_itm_score_v2,
    adapt_t2i_itm_score_v2,
    compute_t2i_itm_score_v2,
    plt_itm_score,
    adapt_i2t_itm_score_v3,
    adapt_t2i_itm_score_v3,
    plt_logging_list,
)
import transformers


import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.clip_vit import create_clip_vit_L
from transformers import BertTokenizer
from tta.tent import softmax_entropy
from lavis.common.registry import registry

from lavis.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import albumentations as A

from tta.datasets import TTA_I2T_Dataset, TTA_T2I_Dataset
from torch.utils.data import DataLoader
from lavis.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.utils.data import DistributedSampler
import numpy as np



def configure_model_blip2_qformer(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    # model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.Qformer.modules(): # 针对blip2模型结构，仅tta更新Qformer参数；freeze visual_encoder/query_tokens/temp(erature)/image_proj/text_proj/itm_head的参数；
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    for m in model.itm_head.modules(): # tta更新itm_head参数；freeze others: visual_encoder/query_tokens/temp(erature)/image_proj/text_proj
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def collect_params_blip2_qformer(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.Qformer.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    for nm, m in model.itm_head.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

import math
from functools import partial
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_rate: float = 0.0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)

def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: Optional[float] = None,
    min_lr_rate: Optional[float] = None,
):
    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`")
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def create_optimizer_scheduler(cfg, model, num_training_steps):
    logging.info(f"create_optimizer_scheduler start")

    model = configure_model_blip2_qformer(model)
    if getattr(cfg.config.tta, "coeffi_exp_temper_is_learnable", False) == True:
        model.coeffi_exp_temper.requires_grad_(True)
    params, param_names = collect_params_blip2_qformer(model)
    if getattr(cfg.config.tta, "coeffi_exp_temper_is_learnable", False) == True:
        params.append(model.coeffi_exp_temper)

    optimizer = torch.optim.AdamW(params=params, lr=cfg.config.tta.init_lr, weight_decay=cfg.config.tta.weight_decay)
    logging.info(f"optimizer {optimizer}")

    lr_scheduler = None
    if getattr(cfg.config.tta, "tta_scheduler", None) == "cosine":
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.config.tta.tta_warmup_ratio * num_training_steps,
            num_training_steps=num_training_steps,
            min_lr = cfg.config.tta.init_lr * 0.1, # 最小学习率
        )     
    logging.info(f"lr_scheduler {lr_scheduler}")

    logging.info(f"create_optimizer_scheduler end")
    return optimizer, lr_scheduler

@torch.no_grad()
def report_metrics(scores_i2t=None, scores_t2i=None, txt2img=None, img2txt=None, prefix_info="", output_dir=""):

    tr1 = -999
    tr5 = -999
    tr10 = -999
    tr_mean = -999
    ir1 = -999
    ir5 = -999
    ir10 = -999
    ir_mean = -999
    r_mean = -999
    # agg_metrics = -999
    mAP_i2t = -999
    mAP_t2i = -999

    if scores_i2t is not None:
        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        average_precisions_i2t = []

        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            relevant_positions = []
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                relevant_positions.append(tmp)
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

            # Calculate average precision
            relevant_positions.sort()
            precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions)]
            average_precision = np.mean(precisions) if precisions else 0
            average_precisions_i2t.append(average_precision)

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        tr_mean = (tr1 + tr5 + tr10) / 3
        # Calculate mAP and format to two decimal places
        mAP_i2t = round(np.mean(average_precisions_i2t) * 100, 2)

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "txt_mAP": mAP_i2t,
        }

    if scores_t2i is not None:
        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])
        average_precisions_t2i = []

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            relevant_positions = []
            tmp = np.where(inds == txt2img[index])[0][0]
            relevant_positions.append(tmp)
            ranks[index] = tmp

            # Calculate average precision
            relevant_positions.sort()
            precisions = [(i + 1) / (pos + 1) for i, pos in enumerate(relevant_positions)]
            average_precision = np.mean(precisions) if precisions else 0
            average_precisions_t2i.append(average_precision)

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        ir_mean = (ir1 + ir5 + ir10) / 3
        mAP_t2i = round(np.mean(average_precisions_t2i) * 100, 2)
        eval_result = {
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "img_mAP": mAP_t2i,
        }

    if scores_i2t is not None and scores_t2i is not None:
        r_mean = (tr_mean + ir_mean) / 2
        # agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "txt_mAP": mAP_i2t,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "img_mAP": mAP_t2i,
            "r_mean": r_mean,
            # "agg_metrics": agg_metrics,
        }

    with open(
        os.path.join(output_dir, "evaluate.txt"), "a"
    ) as f:
        f.write("\n")
        f.write(prefix_info + " " + json.dumps(eval_result) + "\n")
    return eval_result

def plt_logging_list_v2(logging_list, resualt_dir=".", resualt_rank_dir=".", task="i2t", mode="tta"):
    # logging_list.keys() = dict_keys(['index', 'label', 'recall_type', 'sims', 'idxs_i2t', 'score', 'entropy', 'tta_coeffi', 'loss'])
    import numpy as np
    scores = np.array([item["score"] for item in logging_list])
    scores = scores.reshape(-1, scores.shape[-1]) # tta=(5000/25010, 4) eval=(5000/25010, 128)
    entropys = np.array([item["entropy"] for item in logging_list]).mean(axis=-1) #每个batch的entropy均值
   
    if task == "i2t":
        if mode == "tta":
            losses = np.array([item["loss"] for item in logging_list])
            plt.figure(figsize=(32,8))
            plt.plot(losses, alpha=0.7)
            plt.legend(["loss per iteration"])
            plt.savefig(f"{resualt_rank_dir}/{mode}_until_epochs_loss.jpg")
            
            plt.figure(figsize=(32,8))
            plt.plot(scores[:,0], alpha=0.7)
            plt.plot(scores[:,1:].mean(axis=1), alpha=0.7)
            plt.legend(["Positive top1 Score", "Negative Score top6-8 Mean"])
            # plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/{mode}_until_epochs_score_distribution.jpg"))
            plt.savefig(f"{resualt_rank_dir}/{mode}_until_epochs_score_pos_neg.jpg")

            # 计算每10个iter的score均值，输出5000/10维向量
            plt.figure(figsize=(32,8))
            avg_scores_pos = scores[:10*(scores.shape[0]//10),0].reshape(-1, 10).mean(axis=1)
            avg_scores_neg = scores[:10*(scores.shape[0]//10),1:].mean(axis=1).reshape(-1, 10).mean(axis=1)
            plt.plot(avg_scores_pos, marker='o')
            plt.plot(avg_scores_neg, marker='*')
            plt.title("Averaged ITM Score (every 10 samples)")
            plt.xlabel("X (10 samples per X)")
            plt.ylabel("Average Score")
            plt.legend(["Positive top1 Score", "Negative Score top6-8 Mean"])
            # plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/{mode}_until_epochs_avg10_score_distribution.jpg"))
            plt.savefig(f"{resualt_rank_dir}/{mode}_until_epochs_avg10_score_pos_neg.jpg")
        elif mode == "eval":
            # assert False, "i2t eval mode not implemented yet, please use tta mode instead"
            plt.figure(figsize=(32,8))
            plt.plot(scores[:,0:5].mean(axis=1), alpha=0.7)
            plt.plot(scores[:,5:8].mean(axis=1), alpha=0.7)
            plt.legend(["Positive Score top1-5 Mean", "Negative Score top6-8 Mean"])
            # plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/{mode}_until_epochs_score_distribution.jpg"))
            plt.savefig(f"{resualt_rank_dir}/{mode}_until_epochs_score_pos_neg.jpg")

            # 计算每10个iter的score均值，输出5000/10维向量
            plt.figure(figsize=(32,8))
            avg_scores_pos = scores[:10*(scores.shape[0]//10),0:5].mean(axis=1).reshape(-1, 10).mean(axis=1)
            avg_scores_neg = scores[:10*(scores.shape[0]//10),5:8].mean(axis=1).reshape(-1, 10).mean(axis=1)
            plt.plot(avg_scores_pos, marker='o')
            plt.plot(avg_scores_neg, marker='*')
            plt.title("Averaged ITM Score (every 10 samples)")
            plt.xlabel("X (10 samples per X)")
            plt.ylabel("Average Score")
            plt.legend(["Positive Score top1-5 Mean", "Negative Score top6-8 Mean"])
            # plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/{mode}_until_epochs_avg10_score_distribution.jpg"))
            plt.savefig(f"{resualt_rank_dir}/{mode}_until_epochs_avg10_score_pos_neg.jpg")
    elif task == "t2i":
        if mode == "tta":
            plt.figure(figsize=(32,8))
            plt.plot(scores[:,0], alpha=0.7)
            plt.plot(scores[:,1:].mean(axis=1), alpha=0.7)
            plt.legend(["Positive top1 Score", "Negative Score top2-4 Mean"])
            # plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/{mode}_until_epochs_score_distribution.jpg"))
            plt.savefig(f"{resualt_rank_dir}/{mode}_until_epochs_score_pos_neg.jpg")

            # 计算每100个iter的score均值，输出 5000/100维向量
            plt.figure(figsize=(32,8))
            avg_scores_pos = scores[:10*(scores.shape[0]//10),0].reshape(-1, 10).mean(axis=1)
            avg_scores_neg = scores[:10*(scores.shape[0]//10),1:].mean(axis=1).reshape(-1, 10).mean(axis=1)
            plt.plot(avg_scores_pos, marker='o')
            plt.plot(avg_scores_neg, marker='*')
            plt.title("Averaged ITM Score (every 10 samples)")
            plt.xlabel("X (10 samples per X)")
            plt.ylabel("Average Score")
            plt.legend(["Positive top1 Score", "Negative Score top2-4 Mean"])
            # plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/{mode}_until_epochs_avg10_score_distribution.jpg"))
            plt.savefig(f"{resualt_rank_dir}/{mode}_until_epochs_avg10_score_pos_neg.jpg")
        elif mode == "eval":
            assert False, "t2i eval mode not implemented yet, please use tta mode instead"







'''
放到tta_i2t.py main()里面
def test_time_adapt(job_id, cfg, model, tta_dataloader, eval_dataloader, optimizer, scheduler):
    logging.info("test_time_adapt start")
    tta_cfg = cfg.config.tta

    ## path 
    lib_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = lib_root +"/"+ cfg.run_cfg.output_dir +"/"+  job_id
    result_dir = output_dir +"/"+  "result"
    result_rank_dir = result_dir +"/"+  f"rank{get_rank()}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_rank_dir, exist_ok=True)

    ## tta epochs
    logging_list_epochs = []
    eval_logging_list_epochs = []
    for tta_epoch in range(tta_cfg.offline_multi_epochs):
        logging.info(f"start i2t tta epoch {tta_epoch}")
        # score_i2t, itm_score_i2t, logging_list_i2t = adapt_i2t_itm_score_v3(cfg, tta_model.model, dataloader, task_cfg, optimizer, lr_scheduler, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts, tta_epoch)

        score_temper_ = tta_cfg.score_temper if hasattr(tta_cfg, "score_temper") else 1.0

        ## training loop
        logging_list = []
        grad_accum_num = 0

        start_time = time.time()
        model.eval()
        with torch.enable_grad():
            for iter, batch in enumerate(tta_dataloader):

                index = batch["index"]
                sims = batch["sims"]
                idxs_i2t = batch["idxs"]
                image_inputs = batch["image_inputs"].reshape(-1, batch["image_inputs"].size(-2), batch["image_inputs"].size(-1))
                text_ids_inputs = batch["text_ids"].reshape(-1, batch["text_ids"].size(-1))
                text_atts_inputs = batch["text_atts"].reshape(-1, batch["text_atts"].size(-1))
                tta_coeffi = batch["tta_coeffi"] # shape = bs,
                tta_coeffi_proba1 = batch["tta_coeffi_proba1"] # shape = bs,
                tta_coeffi_proba2 = batch["tta_coeffi_proba2"] # shape = bs,
                label = batch["label"] # shape = bs,
                recall_type_2 = batch["recall_type_2"] # list of str  # shape = bs,
                if 0:logging.info("    batch feature end ")

                logits = model.compute_itm_logits(
                    image_inputs=image_inputs.to(model.device), #bs*k_tta,677,1408
                    text_ids=text_ids_inputs.to(model.device), #bs*k_tta,35
                    text_atts=text_atts_inputs.to(model.device), #bs*k_tta,35
                ).float() # logits.shape=bs*k_tta, 2
                logits = logits.reshape(-1, tta_cfg.k_tta, 2) # logits.shape=bs, k_tta, 2
                score = logits[..., 1] # score.shape=bs, k_tta
                if 0:logging.info("    compute_itm_logits end ")  

                ## score = itm_score + cos_sim
                sum_score = score.detach().cpu() + sims.reshape(-1, tta_cfg.k_tta).detach().cpu()

                ## uncertainty
                uncertainty = torch.Tensor([1.]*score.size(0)).to(model.device) # shape = bs,
                if getattr(tta_cfg, "is_uncertainty", False) == True:
                    if getattr(tta_cfg, "uncertainty_type", None) == "kl_itm_itc":
                        uncertainty = nn.functional.kl_div(F.log_softmax(score, dim=-1), F.softmax(sims.reshape(-1, tta_cfg.k_tta).to(model.device), dim=-1), reduction="none").sum(dim=-1) # shape = bs,
                    elif getattr(tta_cfg, "uncertainty_type", None) == "kl_itc_itm":
                        uncertainty = nn.functional.kl_div(F.softmax(sims.reshape(-1, tta_cfg.k_tta).to(model.device), dim=-1), F.log_softmax(score, dim=-1), reduction="none").sum(dim=-1) # shape = bs,
                if 0:logging.info("    uncertainty end ")  

                ## coeffi
                if getattr(tta_cfg, "coeffi_exp_temper_is_learnable", False) == True:
                    tta_coeffi = torch.exp( model.coeffi_exp_temper * (1-(tta_coeffi_proba1+tta_coeffi_proba2)/2) ) 
                else:
                    tta_coeffi = tta_coeffi.to(model.device)
                ## temperature
                score_temper = torch.Tensor([score_temper_]).to(model.device)
                ## adapt
                ### pos/neg softmax entropy
                entropy = -(F.softmax(score * score_temper, dim=-1) * F.log_softmax(score * score_temper, dim=-1)).sum(-1) # score * temper
                ### itm proba sigmoid entropy
                # entropy = (-(F.sigmoid(score * score_temper) * F.log(F.sigmoid(score * score_temper)))).sum(-1)
                loss = entropy
                if getattr(tta_cfg, "is_uncertainty", False) == True:
                    loss = uncertainty * loss
                if getattr(tta_cfg, "top1_match_coeffi", False) == True:
                    if getattr(tta_cfg, "coeffi_exp_temper_is_learnable", False) == True:
                        loss = loss / tta_coeffi + tta_coeffi
                    else:
                        loss = loss / tta_coeffi
                loss = loss.mean()
                loss = loss / tta_cfg.grad_accum_bs
                loss.backward()
                if 0:logging.info("    backward end ")  
                grad_accum_num += 1
                if grad_accum_num >= tta_cfg.grad_accum_bs or iter+1 >= len(tta_dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
                    grad_accum_num = 0
                    if scheduler is not None:
                        scheduler.step()

                ## log_iters logging
                # epoch_entropy_list.append(entropy.mean().detach().cpu().numpy())
                # epoch_loss_list.append(loss.detach().cpu().numpy() * tta_cfg.grad_accum_bs)
                if 1:# if (iter+1) % tta_cfg.log_iters == 0 or iter+1 >= len(tta_dataloader):
                    if scheduler is not None:
                        lr = scheduler.get_last_lr()[0]
                    else:
                        lr = optimizer.param_groups[0]['lr']
                    logging.info(f"[ITM ADAPT rank{get_rank()}] Iteration: {iter}, Iter Entropy Mean: {entropy.mean().detach().cpu().numpy()}, Iter Loss: {loss.detach().cpu().numpy()*tta_cfg.grad_accum_bs}, Learning Rate: {lr}")
               
                ## logging json
                logging_list.append({
                    "index" : index.detach().cpu().numpy().tolist(), # index=ss_idxs[index]
                    "label" : label,
                    "recall_type_2" : recall_type_2,
                    "sims" : sims.reshape(-1, tta_cfg.k_tta).detach().cpu().numpy().tolist(),
                    "idxs_i2t" : idxs_i2t.reshape(-1, tta_cfg.k_tta).detach().cpu().numpy().tolist(),
                    "score" : score.detach().cpu().numpy().tolist(),
                    "sum_score": sum_score.detach().cpu().numpy().tolist(),
                    "entropy" : entropy.detach().cpu().numpy().tolist(),
                    "tta_coeffi": tta_coeffi.detach().cpu().numpy().tolist(),
                    "uncertainty": uncertainty.detach().cpu().numpy().tolist(),
                    "loss" : loss.detach().cpu().numpy() * tta_cfg.grad_accum_bs,
                    "lr": lr,
                })

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("end i2t tta epoch {}, time: {}".format(tta_epoch, total_time_str))

        # # ## torch.distributed
        # score_matrix_i2t[ss_idxs] = score_matrix_i2t_.to(model.device)
        # if dist_utils.is_dist_avail_and_initialized():
        #     dist.barrier()
        #     torch.distributed.all_reduce(
        #         score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        #     )

        # itm_score_list.append(itm_score_i2t)
        logging_list_epochs.extend(logging_list)
        ### plt logging list
        logging.info("plt_logging_list ...")
        plt_logging_list(logging_list_epochs, task="i2t", mode="tta")
        ### save epoch logging
        logging.info("logging_list_json saving ...")
        logging_list_epochs_json = json.dumps(logging_list_epochs)
        json_path = os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/tta_until_epochs_logging_list.json")
        with open(json_path, "w") as f:
            json.dump(logging_list_epochs_json, f)
        # if is_main_process():
        #     ### report metrics
        #     results = report_metrics(scores_i2t=score_i2t, scores_t2i=None, txt2img=None, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics online, epoch {tta_epoch} :")
        #     logging.info(f"report i2t metrics online, epoch {tta_epoch} :")
        #     logging.info(results)
        #     # ### plt & save npy
        #     # npy_path = os.path.join(registry.get_path("output_dir"), f"result/tta_epochs_score_distribution.npy")
        #     # np.save(npy_path, np.concatenate(itm_score_list))
        #     # plt_itm_score(np.concatenate(itm_score_list), task="i2t", mode="tta")

        ## eval
        logging.info("compute i2t itm score offline, epoch %d :", tta_epoch)
        eval_score_i2t, eval_itm_score_i2t, eval_logging_list = compute_i2t_itm_score_v2(model, eval_dataloader, task_cfg, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts, tta_epoch)
        # eval_itm_score_list.append(eval_itm_score_i2t)
        eval_logging_list_epochs.extend(eval_logging_list)
        ### plt logging list
        plt_logging_list(eval_logging_list_epochs, task="i2t", mode="eval")
        ### save epoch logging
        eval_logging_list_epochs_json = json.dumps(eval_logging_list_epochs)
        json_path = os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/eval_until_epochs_logging_list.json")
        with open(json_path, "w") as f:
            json.dump(eval_logging_list_epochs_json, f)
        if is_main_process():
            ### report metrics
            results = report_metrics(scores_i2t=eval_score_i2t, scores_t2i=None, txt2img=tta_dataloader.dataset.txt2img, img2txt=tta_dataloader.dataset.img2txt, prefix_info=f"report i2t metrics offline, epoch {tta_epoch} :")
            logging.info(f"report i2t metrics offline, epoch {tta_epoch} :")
            logging.info(results)
            # ### plt & save npy
            # npy_path = os.path.join(registry.get_path("output_dir"), f"result/eval_epochs_score_distribution.npy")
            # np.save(npy_path, np.concatenate(eval_itm_score_list))
            # plt_itm_score(np.concatenate(eval_itm_score_list), task="i2t", mode="eval")
    
    logging.info("test_time_adapt end")
'''

