"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from lavis.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from lavis.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset
from tta import tent, zhh, itm_adapt
from lavis.runners.runner_base import RunnerBase



@registry.register_runner("runner_tta")
class RunnerTTA(RunnerBase):

    def get_tta_model_optimizer_scheduler(self, model, tta_cfg):

        if tta_cfg.name == "tent":
            model = tent.configure_model_blip2(model)
            params, param_names = tent.collect_params_blip2(model)
            # optimizer = TODO_optimizer(params, lr=1e-3)
            optimizer = torch.optim.AdamW(params=params, lr=tta_cfg.init_lr, weight_decay=tta_cfg.weight_decay)
            tta_model = tent.Tent_Blip2(model, optimizer)
        elif tta_cfg.name == "zhh":
            model = zhh.configure_model_blip2(model)
            params, param_names = zhh.collect_params_blip2(model)
            optimizer = torch.optim.AdamW(params=params, lr=tta_cfg.init_lr, weight_decay=tta_cfg.weight_decay)
            tta_model = zhh.Zhh_Blip2(model, optimizer)
        elif tta_cfg.name == "zhh_topk" or tta_cfg.name == "zhh_topk_ss":
            model = zhh.configure_model_blip2(model)
            params, param_names = zhh.collect_params_blip2(model)
            optimizer = torch.optim.AdamW(params=params, lr=tta_cfg.init_lr, weight_decay=tta_cfg.weight_decay)
            tta_model = zhh.Zhh_Blip2(model, optimizer)
        elif tta_cfg.name == "itm_adapt" or tta_cfg.name == "itm_adapt_ss" or tta_cfg.name == "itm_adapt_sigmoid":
            model = itm_adapt.configure_model_blip2(model)
            params, param_names = itm_adapt.collect_params_blip2(model)
            optimizer = torch.optim.AdamW(params=params, lr=tta_cfg.init_lr, weight_decay=tta_cfg.weight_decay)
            tta_model = itm_adapt.ITM_ADAPT(model, optimizer)
        elif tta_cfg.name == "visenc_itm_adapt":
            model = itm_adapt.configure_visenc_model_blip2(model)
            params, param_names = itm_adapt.collect_visenc_params_blip2(model)
            optimizer = torch.optim.AdamW(params=params, lr=tta_cfg.init_lr, weight_decay=tta_cfg.weight_decay)
            tta_model = itm_adapt.ITM_ADAPT(model, optimizer)
        elif tta_cfg.name == "all_itm_adapt":
            model = itm_adapt.configure_all_model_blip2(model)
            params, param_names = itm_adapt.collect_all_params_blip2(model)
            optimizer = torch.optim.AdamW(params=params, lr=tta_cfg.init_lr, weight_decay=tta_cfg.weight_decay)
            tta_model = itm_adapt.ITM_ADAPT(model, optimizer)
        elif tta_cfg.name in ["itm_adapt_v1","itc_adapt_v1","itm_adapt_v2","itm_adapt_v3"]:
            if tta_cfg.adapt_module == "qformer":
                model = itm_adapt.configure_model_blip2(model)
                params, param_names = itm_adapt.collect_params_blip2(model)
                optimizer = torch.optim.AdamW(params=params, lr=tta_cfg.init_lr, weight_decay=tta_cfg.weight_decay)
                tta_model = itm_adapt.ITM_ADAPT(model, optimizer)
            elif tta_cfg.adapt_module == "visenc_qformer":
                model = itm_adapt.configure_visenc_model_blip2(model)
                params, param_names = itm_adapt.collect_visenc_params_blip2(model)
                optimizer = torch.optim.AdamW(params=params, lr=tta_cfg.init_lr, weight_decay=tta_cfg.weight_decay)
                tta_model = itm_adapt.ITM_ADAPT(model, optimizer)
            elif tta_cfg.adapt_module == "all":
                model = itm_adapt.configure_all_model_blip2(model)
                params, param_names = itm_adapt.collect_all_params_blip2(model)
                optimizer = torch.optim.AdamW(params=params, lr=tta_cfg.init_lr, weight_decay=tta_cfg.weight_decay)
                tta_model = itm_adapt.ITM_ADAPT(model, optimizer)
        
        # lr_scheduler
        lr_scheduler = None
        if getattr(tta_cfg, "tta_scheduler", None) == "cosine":
            num_training_steps = len(sim_matrix_i2t) // tta_cfg.tta_bs * tta_cfg.offline_multi_epochs
            lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer,
                num_warmup_steps=tta_cfg.tta_warmup_ratio * num_training_steps,
                num_training_steps=num_training_steps,
                min_lr = tta_cfg.init_lr * 0.1, # 最小学习率
            )

        return tta_model, optimizer, lr_scheduler

    def evaluate_tta(self, tta_cfg):
    
        dataloader = self.dataloaders.get('test', None)
        assert dataloader, "dataloader for split test is None."

        model = self.unwrap_dist_model(self.model)
        tta_model, optimizer, scheduler = get_tta_model_optimizer_scheduler(model, tta_cfg)

        ## compute embeddings & cosine similarity matrix
        logging.info("compute cosine similarity matrix")
        # if is_main_process():
        #     sim_matrix_i2t, sim_matrix_t2i, image_embeds, vit_feats, text_embeds, text_ids, text_atts = compute_embeds(tta_model.model, dataloader, task_cfg, tta_cfg)
        #     np.save("debugs/debug_sim_matrix_i2t.npy",sim_matrix_i2t.numpy())
        #     np.save("debugs/debug_sim_matrix_t2i.npy",sim_matrix_t2i.numpy())
        #     np.save("debugs/debug_image_embeds.npy",image_embeds.numpy())
        #     np.save("debugs/debug_vit_feats.npy",vit_feats.numpy())
        #     np.save("debugs/debug_text_embeds.npy",text_embeds.numpy())
        #     np.save("debugs/debug_text_ids.npy",text_ids.numpy())
        #     np.save("debugs/debug_text_atts.npy",text_atts.numpy())
        sim_matrix_i2t = torch.from_numpy(np.load("debugs/debug_sim_matrix_i2t.npy"))
        sim_matrix_t2i = torch.from_numpy(np.load("debugs/debug_sim_matrix_t2i.npy"))
        image_embeds = torch.from_numpy(np.load("debugs/debug_image_embeds.npy"))
        vit_feats = torch.from_numpy(np.load("/data/jiahao/blip2_embeddings/debug_vit_feats.npy"))
        text_embeds = torch.from_numpy(np.load("debugs/debug_text_embeds.npy"))
        text_ids = torch.from_numpy(np.load("debugs/debug_text_ids.npy"))
        text_atts = torch.from_numpy(np.load("debugs/debug_text_atts.npy"))
        if is_main_process():
            result = report_metrics(scores_i2t=sim_matrix_i2t.numpy(), scores_t2i=sim_matrix_t2i.numpy(), txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report recall metrics, zero-shot :")
            logging.info(f"report recall metrics, zero-shot :")
            logging.info(result)

        ## i2t tta
        if tta_cfg.tta_task == "i2t":
            ## zero_shot
            if tta_cfg.zero_shot_eval:
                logging.info("compute i2t itm score, zero-shot")
                score_i2t_zeroshot, itm_score_i2t_zeroshot, eval_logging_list_i2t_ze = compute_i2t_itm_score_v2(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts)
                if is_main_process():
                    result = report_metrics(scores_i2t=score_i2t_zeroshot, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics, zero-shot :")
                    logging.info(f"report i2t metrics, zero-shot :")
                    logging.info(result)


                    
            logging.info(f"lr_scheduler {lr_scheduler}")

            ## multi epochs
            logging_list_i2t_list = []
            eval_logging_list_i2t_list = []
            for tta_epoch in range(tta_cfg.offline_multi_epochs):
                logging.info(f"start i2t tta epoch {tta_epoch}")
                ## tta
                logging.info("adapt i2t itm score online, epoch %d :", tta_epoch)
                score_i2t, itm_score_i2t, logging_list_i2t = adapt_i2t_itm_score_v3(cfg, tta_model.model, dataloader, task_cfg, optimizer, lr_scheduler, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts, tta_epoch)





                
                logging_list_i2t_list.extend(logging_list_i2t)
                ### plt logging list
                plt_logging_list(logging_list_i2t_list, task="i2t", mode="tta")
                ### save epoch logging
                logging_list_json = json.dumps(logging_list_i2t_list)
                json_path = os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/tta_until_epochs_logging_list.json")
                with open(json_path, "w") as f:
                    json.dump(logging_list_json, f)
                if is_main_process():
                    ### report metrics
                    results = report_metrics(scores_i2t=score_i2t, scores_t2i=None, txt2img=None, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics online, epoch {tta_epoch} :")
                    logging.info(f"report i2t metrics online, epoch {tta_epoch} :")
                    logging.info(results)

                ## eval
                logging.info("compute i2t itm score offline, epoch %d :", tta_epoch)
                score_i2t, eval_itm_score_i2t, eval_logging_list_i2t = compute_i2t_itm_score_v2(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts, tta_epoch)
                eval_logging_list_i2t_list.extend(eval_logging_list_i2t)
                ### plt logging list
                plt_logging_list(eval_logging_list_i2t_list, task="i2t", mode="eval")
                ### save epoch logging
                logging_list_json = json.dumps(eval_logging_list_i2t_list)
                json_path = os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/eval_until_epochs_logging_list.json")
                with open(json_path, "w") as f:
                    json.dump(logging_list_json, f)
                if is_main_process():
                    ### report metrics
                    results = report_metrics(scores_i2t=score_i2t, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics offline, epoch {tta_epoch} :")
                    logging.info(f"report i2t metrics offline, epoch {tta_epoch} :")
                    logging.info(results)
