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



from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from tta.tta_datasets import preprocess_tta_dataset, create_tta_dataset, create_eval_dataset, create_eval_dataloader, create_tta_dataloader
from tta.utils import compute_embeds, compute_i2t_itm_score_v2, plt_logging_list
from tta.tta_utils import create_optimizer_scheduler, report_metrics, plt_logging_list_v2#, test_time_adapt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import logging
import time
import datetime
import json

from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A



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

    #### Path 
    lib_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = lib_root +"/"+ cfg.run_cfg.output_dir +"/"+  job_id
    logging.info(f"\nCreate output_dir: {output_dir} ...")
    result_dir = output_dir +"/"+  "result"
    result_rank_dir = result_dir +"/"+  f"rank{get_rank()}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_rank_dir, exist_ok=True)

    #### Blip2Qformer
    logging.info("\nInitialize model ...")
    model = Blip2Qformer.from_config(cfg=cfg.model_cfg)

    #### eval_dataset
    logging.info("\nInitialize dataset ...")
    eval_dataset = create_eval_dataset(cfg)
    eval_dataloader = create_eval_dataloader(cfg, eval_dataset)

    #### tta_dataset
    logging.info("\nInitialize tta feature ...")
    # sims_matrix_i2t, sims_matrix_t2i, image_embeds, vit_feats, text_embeds, text_ids, text_atts = compute_embeds(model, eval_dataloader, cfg.run_cfg, cfg.tta_cfg)

    #### pre-load features
    # np.save("debugs/debug_sim_matrix_i2t.npy",sims_matrix_i2t.numpy())
    # np.save("debugs/debug_sim_matrix_t2i.npy",sims_matrix_t2i.numpy())
    # np.save("debugs/debug_image_embeds.npy",image_embeds.numpy())
    # np.save("/data/jiahao/blip2_embeddings/debug_vit_feats.npy",vit_feats.numpy())
    # np.save("debugs/debug_text_embeds.npy",text_embeds.numpy())
    # np.save("debugs/debug_text_ids.npy",text_ids.numpy())
    # np.save("debugs/debug_text_atts.npy",text_atts.numpy())
    if is_main_process():
        logging.warning(f"get_rank:{get_rank()} load debug_sim_matrix_i2t.npy")
        sims_matrix_i2t = torch.from_numpy(np.load("debugs/debug_sim_matrix_i2t.npy"))
        
        # sims_matrix_t2i = torch.from_numpy(np.load("debugs/debug_sim_matrix_t2i.npy"))
        # image_embeds = torch.from_numpy(np.load("debugs/debug_image_embeds.npy"))

        logging.warning(f"get_rank:{get_rank()} load debug_vit_feats.npy")
        vit_feats = torch.from_numpy(np.load("/data/jiahao/blip2_embeddings/debug_vit_feats.npy"))
       
        # text_embeds = torch.from_numpy(np.load("debugs/debug_text_embeds.npy"))
        
        logging.warning(f"get_rank:{get_rank()} load debug_text_ids.npy")
        text_ids = torch.from_numpy(np.load("debugs/debug_text_ids.npy"))
        

        logging.warning(f"get_rank:{get_rank()} load debug_text_atts.npy")
        text_atts = torch.from_numpy(np.load("debugs/debug_text_atts.npy"))
        
        logging.warning(f"get_rank:{get_rank()} preprocess_tta_dataset")
        ss_idxs = preprocess_tta_dataset(cfg, eval_dataloader.dataset.img2txt, sims_matrix_i2t, vit_feats, text_ids, text_atts)

    # # ## broadcast features to all process
    # if is_dist_avail_and_initialized():
        
    #     if is_main_process():
    #         sims_matrix_i2t = sims_matrix_i2t.cuda()
    #     else:
    #         sims_matrix_i2t = torch.empty([5000, 25010], dtype=torch.float32,device='cuda')
    #     dist.broadcast(sims_matrix_i2t, src=0)
    #     sims_matrix_i2t = sims_matrix_i2t.cpu()

    #     if is_main_process():
    #         vit_feats = vit_feats.cuda()
    #     else:
    #         vit_feats = torch.empty([5000, 677, 1408], dtype=torch.float32,device='cuda')
    #     dist.broadcast(vit_feats, src=0)
    #     vit_feats = vit_feats.cpu()
            
    #     if is_main_process():
    #         text_ids = text_ids.cuda()
    #     else:
    #         text_ids = torch.empty([25010, 35], dtype=torch.int64,device='cuda')
    #     dist.broadcast(text_ids, src=0)
    #     text_ids = text_ids.cpu()
            
    #     if is_main_process():
    #         text_atts = text_atts.cuda()
    #     else:
    #         text_atts = torch.empty([25010, 35], dtype=torch.int64,device='cuda')
    #     dist.broadcast(text_atts, src=0)
    #     text_atts = text_atts.cpu()
        
    # sims_matrix_i2t, vit_feats, text_ids, text_atts = sims_matrix_i2t.cpu(), vit_feats.cpu(), text_ids.cpu(), text_atts.cpu()

    logging.info("\nInitialize tta dataset ...")
    tta_dataset = create_tta_dataset(cfg, ss_idxs)
    tta_dataloader, tta_sampler = create_tta_dataloader(cfg, tta_dataset)
    
    logging.info("\nInitialize optimizer ...")
    num_training_steps = len((tta_dataloader)) * cfg.config.tta.offline_multi_epochs
    optimizer, scheduler = create_optimizer_scheduler(cfg, model, num_training_steps)

    logging.info("\nStart test_time_adapt ...")
    # test_time_adapt(job_id, cfg, model, tta_dataloader, eval_dataloader, optimizer, scheduler)

     ## model.to(device)
    if is_dist_avail_and_initialized():
        local_rank = get_rank()
        device = torch.device(cfg.run_cfg.device, local_rank)
    else:
        device =  torch.device(cfg.run_cfg.device)
    model = model.to(device)
    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[local_rank]).module

    ## zero_shot eval
    if cfg.config.tta.zero_shot_eval:
        logging.info("compute i2t itm score, zero-shot")
        score_i2t_zeroshot, itm_score_i2t_zeroshot, eval_logging_list_i2t_ze = compute_i2t_itm_score_v2(model, eval_dataloader, cfg.run_cfg, cfg.tta_cfg, sims_matrix_i2t, vit_feats, text_ids, text_atts,  epoch=-1, result_rank_dir=result_rank_dir)
        result = report_metrics(scores_i2t=score_i2t_zeroshot, scores_t2i=None, txt2img=eval_dataloader.dataset.txt2img, img2txt=eval_dataloader.dataset.img2txt, prefix_info=f"rank{get_rank()} report i2t metrics, zero-shot :")
        logging.info(f"report i2t metrics, zero-shot :")
        logging.info(result)
    
    logging.info("============= test_time_adapt start =============")
    tta_cfg = cfg.config.tta
    score_temper_ = tta_cfg.score_temper if hasattr(tta_cfg, "score_temper") else 1.0

    ## tta epochs
    logging_list_epochs = []
    eval_logging_list_epochs = []
    for tta_epoch in range(tta_cfg.offline_multi_epochs):
        logging.info(f"start i2t tta epoch {tta_epoch}")
        # score_i2t, itm_score_i2t, logging_list_i2t = adapt_i2t_itm_score_v3(cfg, tta_model.model, dataloader, task_cfg, optimizer, lr_scheduler, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts, tta_epoch)

        ## training loop
        logging_list = []
        grad_accum_num = 0

        start_time = time.time()
        model.eval()
        if is_dist_avail_and_initialized() and tta_sampler is not None:
            tta_sampler.set_epoch(tta_epoch)
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
                    tta_coeffi = torch.exp( model.coeffi_exp_temper * (1-(tta_coeffi_proba1.to(model.device)+tta_coeffi_proba2.to(model.device))/2) ).to(model.device) 
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
                    logging.info(f"[ITM ADAPT rank{get_rank()}] Iteration: {iter}, Iter Entropy Mean: {entropy.mean().detach().cpu().numpy()}, Iter Loss: {loss.detach().cpu().numpy()*tta_cfg.grad_accum_bs}, Learning Rate: {lr}, model.coeffi_exp_temper: {model.coeffi_exp_temper}")
               
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
                    "coeffi_exp_temper": model.coeffi_exp_temper.detach().cpu().numpy().tolist(),
                    "learnable_empty_embedding": model.learnable_empty_embedding.detach().cpu().numpy().tolist(),
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
        plt_logging_list_v2(logging_list_epochs, result_dir, result_rank_dir, task="i2t", mode="tta")
        ### save epoch logging
        logging.info("logging_list_json saving ...")
        logging_list_epochs_json = json.dumps(logging_list_epochs)
        json_path = f"{result_rank_dir}/tta_until_epochs_logging_list.json"
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

        # torch.cuda.empty_cache()
        # gc.collect()
        ## eval
        logging.info("compute i2t itm score offline, epoch %d :", tta_epoch)

        # logging.warning(f"get_rank:{get_rank()} load debug_sim_matrix_i2t.npy")
        # sims_matrix_i2t = torch.from_numpy(np.load("debugs/debug_sim_matrix_i2t.npy"))
        
        # # sims_matrix_t2i = torch.from_numpy(np.load("debugs/debug_sim_matrix_t2i.npy"))
        # # image_embeds = torch.from_numpy(np.load("debugs/debug_image_embeds.npy"))

        # logging.warning(f"get_rank:{get_rank()} load debug_vit_feats.npy")
        # vit_feats = torch.from_numpy(np.load("/data/jiahao/blip2_embeddings/debug_vit_feats.npy"))
       
        # # text_embeds = torch.from_numpy(np.load("debugs/debug_text_embeds.npy"))
        
        # logging.warning(f"get_rank:{get_rank()} load debug_text_ids.npy")
        # text_ids = torch.from_numpy(np.load("debugs/debug_text_ids.npy"))
        

        # logging.warning(f"get_rank:{get_rank()} load debug_text_atts.npy")
        # text_atts = torch.from_numpy(np.load("debugs/debug_text_atts.npy"))

        eval_score_i2t, eval_itm_score_i2t, eval_logging_list = compute_i2t_itm_score_v2(model, eval_dataloader, cfg.run_cfg, tta_cfg, sims_matrix_i2t, vit_feats, text_ids, text_atts, tta_epoch, result_rank_dir)
        # eval_itm_score_list.append(eval_itm_score_i2t)
        eval_logging_list_epochs.extend(eval_logging_list)
        ### plt logging list
        plt_logging_list_v2(eval_logging_list_epochs, result_dir, result_rank_dir, task="i2t", mode="eval")
        ### save epoch logging
        eval_logging_list_epochs_json = json.dumps(eval_logging_list_epochs)
        json_path = f"{result_rank_dir}/eval_until_epochs_logging_list.json"
        with open(json_path, "w") as f:
            json.dump(eval_logging_list_epochs_json, f)
        if is_main_process():
            ### report metrics
            results = report_metrics(scores_i2t=eval_score_i2t, scores_t2i=None, txt2img=eval_dataloader.dataset.txt2img, img2txt=eval_dataloader.dataset.img2txt, prefix_info=f"report i2t metrics offline, epoch {tta_epoch} :", output_dir=output_dir)
            logging.info(f"report i2t metrics offline, epoch {tta_epoch} :")
            logging.info(results)
            # ### plt & save npy
            # npy_path = os.path.join(registry.get_path("output_dir"), f"result/eval_epochs_score_distribution.npy")
            # np.save(npy_path, np.concatenate(eval_itm_score_list))
            # plt_itm_score(np.concatenate(eval_itm_score_list), task="i2t", mode="eval")
    
    logging.info("============= test_time_adapt end =============")


if __name__ == "__main__":
    main()



# command
# CUDA_VISIBLE_DEVICES=1 nohup python tta_i2t.py --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.yaml --is_tta True > ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1,2 nohup python -m torch.distributed.run --nproc_per_node=2 tta_i2t.py --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.yaml --is_tta True > ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.out 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 tta_i2t.py --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.yaml --is_tta True > ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.out 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 tta_i2t.py --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.yaml --is_tta True

# debug
# --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp_debug.yaml --is_tta True
# --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.yaml --is_tta True





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
