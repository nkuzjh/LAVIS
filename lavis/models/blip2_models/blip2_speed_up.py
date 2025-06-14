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

from tqdm import tqdm
import matplotlib.pyplot as plt
import json


@torch.no_grad()
def compute_sim_matrix_spd(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    model.eval()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds: # 5000,32,256
        sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
        sim_i2t, _ = sim_q2t.max(0) #32,25010
        sims_matrix.append(sim_i2t) # 25010
    sims_matrix = torch.stack(sims_matrix, dim=0) #5000,25010

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim
        if i % 50 == 0:
            logging.info("[ITM Evaluation]")

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), sims_matrix.cpu().numpy()


def find_recall_type(label_list, topk_idx):
    if topk_idx[0] in label_list:
        return "recall@1"
    else: 
        for _idx in topk_idx[:5].tolist():
            if _idx in label_list:
                return "recall@5"
        for _idx in topk_idx[:10].tolist():
            if _idx in label_list:
                return "recall@10"
    return "negative sample"

def compute_i2t_sim_matrix_adapt_itm(model, data_loader, optimizer, tta_cfg, epoch, **kwargs):
    k_test = kwargs.pop("k_test")

    logging.info("i2t online Evaluation...")

    logging.info("Stage 1: cos_sim Recall")
    model.eval()
    with torch.no_grad():
        logging.info("    text features...")
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        start_time = time.time()
        for i in tqdm(range(0, num_text, text_bs)):
            # print("\r text_batch_i: ", i, end="")
            text = texts[i : min(num_text, i + text_bs)]
            text_input = model.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(model.device)
            text_feat = model.forward_text(text_input)
            text_embed = F.normalize(model.text_proj(text_feat))
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)

        curr_time = time.time() - start_time
        logging.info("  text features time {}".format(str(datetime.timedelta(seconds=int(curr_time)))))

        logging.info("    image features...")
        vit_feats = []
        image_embeds = []
        start_time = time.time()
        for i, samples in  tqdm(enumerate(data_loader), total=len(data_loader)):
            # print("\r image_batch_i: ", i*len(samples["image"]), end="")
            image = samples["image"]

            image = image.to(model.device)
            image_feat, vit_feat = model.forward_image(image)
            image_embed = model.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)

            vit_feats.append(vit_feat.cpu())
            image_embeds.append(image_embed)

        vit_feats = torch.cat(vit_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        curr_time = time.time() - start_time
        logging.info("  image features time {}".format(str(datetime.timedelta(seconds=int(curr_time)))))

        logging.info("    similarity matrix...") # 这里F.normaliza()后的矩阵相乘@ 就是cosine_similarity
        sims_matrix = []
        start_time = time.time()
        for image_embed in tqdm(image_embeds): # 5000,32,256 # 使用循环避免显存oom
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010 -> 25010
            # sim_i2t = sim_i2t / model.temp
            sims_matrix.append(sim_i2t)
        sims_matrix_i2t = torch.stack(sims_matrix, dim=0) #5000,25010
        sims_matrix_t2i = sims_matrix_i2t.t()

        curr_time = time.time() - start_time
        logging.info("  similarity matrix time {}".format(str(datetime.timedelta(seconds=int(curr_time)))))

        labels = data_loader.dataset.img2txt

    logging.info("Stage 2: itm Rerank ")

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)
    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_i2t.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_i2t.size(0), start + step)
    if hasattr(tta_cfg, 'grad_accum_bs'):
        itm_loss_backward_accum_bs = tta_cfg.grad_accum_bs #1 #64
    else:
        itm_loss_backward_accum_bs = 64 #1 #64
    if hasattr(tta_cfg, 'temper'):
        sigmoid_temper = tta_cfg.temper
    else:
        sigmoid_temper = 1.
    grad_accum_num = 0

    iters_entropy = 0.0
    epoch_entropy_list = []
    iters_loss = 0.0
    epoch_loss_list = []
    coeffi_list = []

    start_time = time.time()
    model.train()
    with tqdm( total=sims_matrix_i2t.size(0)) as tbar:
        for i, sims_i2t in enumerate(sims_matrix_i2t[start:end]): # 遍历每个image与25010个text的sim_matrix
            topk_sim_i2t, topk_idx_i2t = sims_i2t.topk(k=k_test, dim=0) #sims.shape=25010 topk_sim.shape=128 topk_idx=top128_idx
            image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device) # vit_feats[i].shape=1,677,1408 image_inputs.shape=128,677,1408
            score = model.compute_itm(
                image_inputs=image_inputs, #128,677,1408
                text_ids=text_ids[topk_idx_i2t], #128,35
                text_atts=text_atts[topk_idx_i2t], #128,35
            ).float() # score.shape=128

            recall_type = find_recall_type(labels[i], topk_idx_i2t)
            if recall_type=="negative sample": #i==112
                a=999

            top1_match_coeffi = torch.ones(1).to(model.device)
            if hasattr(tta_cfg, 'top1_match_coeffi') and tta_cfg.top1_match_coeffi == True:
                if hasattr(tta_cfg, 'top1_match_coeffi_src') and tta_cfg.top1_match_coeffi_src == "sigmoid":
                    proba_top1_sim_i2t = F.sigmoid(topk_sim_i2t)[0]
                else:
                    proba_top1_sim_i2t = F.softmax(topk_sim_i2t, dim=0)[0]
                proba_sim_t2i_top1_idx_i2t = torch.zeros(1).to(proba_top1_sim_i2t.device)
                topk_sim_t2i_top1_idx_i2t, topk_idx_t2i_top1_idx_i2t = sims_matrix_t2i[topk_idx_i2t[0]].topk(k=k_test, dim=0)
                if i in topk_idx_t2i_top1_idx_i2t:
                    idx_i_in_topk_idx_t2i_top1_idx_i2t = torch.where(topk_idx_t2i_top1_idx_i2t == i)
                    if hasattr(tta_cfg, 'top1_match_coeffi_src') and tta_cfg.top1_match_coeffi_src == "sigmoid":
                        proba_sim_t2i_top1_idx_i2t = F.sigmoid(topk_sim_t2i_top1_idx_i2t)[idx_i_in_topk_idx_t2i_top1_idx_i2t]
                    else:
                        proba_sim_t2i_top1_idx_i2t = F.softmax(topk_sim_t2i_top1_idx_i2t, dim=0)[idx_i_in_topk_idx_t2i_top1_idx_i2t]
                top1_match_coeffi = torch.exp(1 - (proba_top1_sim_i2t + proba_sim_t2i_top1_idx_i2t)/2 )

            # itm entropy adapt
            if hasattr(tta_cfg, 'entropy_type') and tta_cfg.entropy_type == "sigmoid_sum":
                loss_entropy_topk_gallery = -(F.sigmoid(score) * torch.log(F.sigmoid(score))).sum() / sigmoid_temper
            elif hasattr(tta_cfg, 'entropy_type') and tta_cfg.entropy_type == "sigmoid_mean":
                loss_entropy_topk_gallery = -(F.sigmoid(score) * torch.log(F.sigmoid(score))).mean() / sigmoid_temper
            elif hasattr(tta_cfg, 'entropy_type') and tta_cfg.entropy_type == "softmax":
                loss_entropy_topk_gallery = -(F.softmax(score, dim=0) * F.log_softmax(score, dim=0)).sum()
            else:
                loss_entropy_topk_gallery = -(F.softmax(score, dim=0) * F.log_softmax(score, dim=0)).sum()
            loss_entropy_uncertainty = loss_entropy_topk_gallery.mean() / top1_match_coeffi
            loss_entropy_uncertainty = loss_entropy_uncertainty / itm_loss_backward_accum_bs
            loss_entropy_uncertainty.backward()
            grad_accum_num += 1
            if grad_accum_num >= itm_loss_backward_accum_bs or i>=end-1:
                optimizer.step()
                optimizer.zero_grad()
                grad_accum_num = 0

            #score = itm_score + cos_sim
            score_matrix_i2t[start+i, topk_idx_i2t] = score + topk_sim_i2t

            ## tqdm logging
            tbar.set_postfix(entropy=loss_entropy_topk_gallery, loss=loss_entropy_uncertainty*itm_loss_backward_accum_bs, lr=optimizer.param_group['lr'])
            ## log_iters logging
            iters_entropy += loss_entropy_topk_gallery.mean().detach().cpu().numpy()
            iters_loss += loss_entropy_uncertainty.detach().cpu().numpy() * itm_loss_backward_accum_bs
            coeffi_list.append({
                "label" : labels[i],
                "score[:10]" : score[:10].detach().cpu().numpy().tolist(),
                "recall_type" : recall_type,
                "entropy" : loss_entropy_topk_gallery.detach().cpu().numpy().tolist(),
                "topk_sim_i2t[:10]" : topk_sim_i2t[:10].detach().cpu().numpy().tolist(),
                "topk_idx_i2t[:10]" : topk_idx_i2t[:10].detach().cpu().numpy().tolist(),
                "proba_top1_sim_i2t" : proba_top1_sim_i2t.detach().cpu().numpy().tolist(),
                "topk_sim_t2i_top1_idx_i2t[:10]" : topk_sim_t2i_top1_idx_i2t[:10].detach().cpu().numpy().tolist(),
                "topk_idx_t2i_top1_idx_i2t[:10]" : topk_idx_t2i_top1_idx_i2t[:10].detach().cpu().numpy().tolist(),
                "proba_sim_t2i_top1_idx_i2t" : proba_sim_t2i_top1_idx_i2t.detach().cpu().numpy().tolist(),
                "top1_match_coeffi": top1_match_coeffi.detach().cpu().numpy().tolist(),
                "loss_entropy_uncertainty" : [ loss * itm_loss_backward_accum_bs for loss in loss_entropy_uncertainty.detach().cpu().numpy().tolist()],
            })
            if i % tta_cfg.log_iters == 0 or i>= end:
                logging.info(" ")
                logging.info(f"[i2t online Evaluation itm adapt] Iteration: {i}, Iters Average Entropy: {iters_entropy / tta_cfg.log_iters}, Iters Average Loss: {iters_loss / tta_cfg.log_iters} ")
                iters_entropy = 0.0
                iters_loss = 0.0
                epoch_entropy_list.append(loss_entropy_topk_gallery.mean().detach().cpu().numpy())
                epoch_loss_list.append(loss_entropy_uncertainty.detach().cpu().numpy() * itm_loss_backward_accum_bs)

            # if tta_cfg.debug_visual == True:
            #     logging.info(f"    label : {labels[i]}")
            #     logging.info(f"    score[:10] : {score[:10]}")
            #     logging.info(f"    recall_type : {recall_type}")
            #     logging.info(f"    entropy : {loss_entropy_topk_gallery}")

            #     logging.info(f"    topk_sim_i2t[:10] : {topk_sim_i2t[:10]}")
            #     logging.info(f"    topk_idx_i2t[:10] : {topk_idx_i2t[:10]}")
            #     logging.info(f"    proba_top1_sim_i2t : {proba_top1_sim_i2t}")

            #     logging.info(f"    topk_sim_t2i_top1_idx_i2t[:10] : {topk_sim_t2i_top1_idx_i2t[:10]}")
            #     logging.info(f"    topk_idx_t2i_top1_idx_i2t[:10] : {topk_idx_t2i_top1_idx_i2t[:10]}")
            #     logging.info(f"    proba_sim_t2i_top1_idx_i2t : {proba_sim_t2i_top1_idx_i2t}")

            #     logging.info(f"    top1_match_coeffi : {top1_match_coeffi}")
            #     logging.info(f"    loss_entropy_uncertainty : {loss_entropy_uncertainty * itm_loss_backward_accum_bs}")

    coeffi_list_json = json.dumps(coeffi_list)
    json_path = os.path.join(registry.get_path("output_dir"), f"epoch{epoch}_coeffi_list.json")
    with open(json_path, "w") as f:
        json.dump(coeffi_list_json, f)

    plt.figure()
    plt.plot(epoch_entropy_list) 
    plt.savefig(os.path.join(registry.get_path("output_dir"), f"epoch{epoch}_entropy.jpg"))
    plt.figure()
    plt.plot(epoch_loss_list) 
    plt.savefig(os.path.join(registry.get_path("output_dir"), f"epoch{epoch}_loss.jpg"))
    # if tta_cfg.debug_visual == True:
    #     plt.show()

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("i2t online Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().detach().numpy()


