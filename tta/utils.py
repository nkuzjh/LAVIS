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


def find_recall_type(label_list, topk_idx):
    label_list = label_list if isinstance(label_list, list) else [label_list]
    if topk_idx[0] in label_list:
        return "recall@1"
    else:
        for _idx in topk_idx[1:5].tolist():
            if _idx in label_list:
                return "recall@5"
        for _idx in topk_idx[5:10].tolist():
            if _idx in label_list:
                return "recall@10"
    return "negative sample"

def find_recall_types(labels, sims_matrix_i2t, k_test):
    recall_types = []
    for i, sims_i2t in enumerate(sims_matrix_i2t):
        label_list = labels[i]
        topk_sim, topk_idx = sims_i2t.topk(k=k_test, dim=0)
        recall_types.append(find_recall_type(label_list, topk_idx))
    return recall_types

def compute_kl_divergence(true_proba, pred_proba):
    return [ F.kl_div(t.log(), p) for t, p in zip(true_proba, pred_proba) ]

def compute_kl_coeffis(sims_mat_i2t, sims_mat_i2t_aug):
    coeffis_i2t, coeffis_t2i = [], []
    # KL散度越小说明分布越相似，即更可信,所以使用exp(-Dkl)作为加权系数让可信样本系数趋近于1，不可信样本系数趋近于0
    coeffis_i2t = compute_kl_divergence(sims_mat_i2t, sims_mat_i2t_aug)
    coeffis_t2i = compute_kl_divergence(sims_mat_i2t.t(), sims_mat_i2t_aug.t())

    return coeffis_i2t, coeffis_t2i

def compute_top1_match_coeffis(sims_matrix_i2t, sims_matrix_t2i, top1_match_coeffi_src, k_test):
    coeffis = []
    for i, sims_i2t in enumerate(sims_matrix_i2t):
        topk_sim_i2t, topk_idx_i2t = sims_i2t.topk(k=k_test, dim=0)

        coeffi = torch.ones(1)
        if top1_match_coeffi_src == "sigmoid":
            proba_top1_sim_i2t = F.sigmoid(topk_sim_i2t)[0]
        elif top1_match_coeffi_src == "softmax":
            proba_top1_sim_i2t = F.softmax(topk_sim_i2t, dim=0)[0]
        elif top1_match_coeffi_src == "cos_sim":
            proba_top1_sim_i2t = topk_sim_i2t[0]
        else:
            proba_top1_sim_i2t = topk_sim_i2t[0]
        proba_sim_t2i_top1_idx_i2t = torch.zeros(1)
        topk_sim_t2i_top1_idx_i2t, topk_idx_t2i_top1_idx_i2t = sims_matrix_t2i[topk_idx_i2t[0]].topk(k=k_test, dim=0)
        if i in topk_idx_t2i_top1_idx_i2t:
            idx_i_in_topk_idx_t2i_top1_idx_i2t = torch.where(topk_idx_t2i_top1_idx_i2t == i)
            if top1_match_coeffi_src == "sigmoid":
                proba_sim_t2i_top1_idx_i2t = F.sigmoid(topk_sim_t2i_top1_idx_i2t)[idx_i_in_topk_idx_t2i_top1_idx_i2t]
            elif top1_match_coeffi_src == "softmax":
                proba_sim_t2i_top1_idx_i2t = F.softmax(topk_sim_t2i_top1_idx_i2t, dim=0)[idx_i_in_topk_idx_t2i_top1_idx_i2t]
            elif top1_match_coeffi_src == "cos_sim":
                proba_sim_t2i_top1_idx_i2t = topk_sim_t2i_top1_idx_i2t[idx_i_in_topk_idx_t2i_top1_idx_i2t]
            else:
                proba_sim_t2i_top1_idx_i2t = topk_sim_t2i_top1_idx_i2t[idx_i_in_topk_idx_t2i_top1_idx_i2t]
        coeffi = torch.exp( 1 - (proba_top1_sim_i2t + proba_sim_t2i_top1_idx_i2t)/2 )
        coeffis.append(coeffi)
    return coeffis

def get_image_transform():
    tta_image_transforms = A.Compose([
        # A.RandomResizedCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        # A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        # A.ToTensorV2(),
    ])
    return tta_image_transforms

def compute_embeds(model, dataloader, task_cfg, tta_cfg, is_aug=False):
    logging.info("compute_embeds: start")

    model.eval()
    with torch.no_grad():
        logging.info("    text features...")
        start_time = time.time()
        texts = dataloader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in tqdm(range(0, num_text, text_bs), total=num_text // text_bs):
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
        logging.info("    text features time: {}".format(str(datetime.timedelta(seconds=int(curr_time)))))

        logging.info("    image features...")
        if is_aug and tta_cfg.top1_match_coeffi==True and tta_cfg.top1_match_coeffi_src in ["KL_itc", "KL_itm"] and tta_cfg.KL_aug_modal in ["image","multimodal"]:
            image_transform = get_image_transform()
        else:
            image_transform = None
        start_time = time.time()
        vit_feats = []
        image_embeds = []
        for i, samples in  tqdm(enumerate(dataloader), total=len(dataloader)):
            image = samples["image"]
            if is_aug and image_transform is not None:
                imgs = []
                # i = 0
                for img in image.cpu().numpy().transpose(0,2,3,1):
                    # from PIL import Image
                    # import numpy as np
                    # print(img.shape)
                    # orig_img = Image.fromarray((255*(img - img.min()) / (img.max()-img.min())).astype(np.uint8))
                    # orig_img.save(f"orig_img_{i}.jpg")
                    _img = image_transform(image=img)["image"]
                    # aug_img = Image.fromarray((255*(_img - _img.min()) / (_img.max()-_img.min())).astype(np.uint8))
                    # aug_img.save(f"aug_img_{i}.jpg")
                    # i+=1
                    imgs.append(torch.from_numpy(_img.transpose(2,0,1)))
                image = torch.stack(imgs)
            image = image.to(model.device)
            image_feat, vit_feat = model.forward_image(image)
            image_embed = model.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)

            vit_feats.append(vit_feat.cpu())
            image_embeds.append(image_embed)

        vit_feats = torch.cat(vit_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        curr_time = time.time() - start_time
        logging.info("    image features time {}".format(str(datetime.timedelta(seconds=int(curr_time)))))

        logging.info("    similarity matrix...") # 这里F.normaliza()后的矩阵相乘@ 就是cosine_similarity
        start_time = time.time()
        sims_matrix = []
        for image_embed in tqdm(image_embeds, total=len(image_embeds)): # 5000,32,256 # 使用循环避免显存oom
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010 -> 25010
            # sim_i2t = sim_i2t / model.temp
            sims_matrix.append(sim_i2t)
        sims_matrix_i2t = torch.stack(sims_matrix, dim=0) #5000,25010
        sims_matrix_t2i = sims_matrix_i2t.t()

        curr_time = time.time() - start_time
        logging.info("  similarity matrix time {}".format(str(datetime.timedelta(seconds=int(curr_time)))))

    logging.info("compute_embeds: end")
    return sims_matrix_i2t.detach().cpu(), sims_matrix_t2i.detach().cpu(), image_embeds.detach().cpu(), vit_feats.detach().cpu(), text_embeds.detach().cpu(), text_ids.detach().cpu(), text_atts.detach().cpu()

def compute_i2t_itm_score(model, dataloader, task_cfg, tta_cfg, sims_matrix_i2t, vit_feats, text_ids, text_atts, epoch=-1):
    logging.info("compute_i2t_itm_score: start")

    k_test = task_cfg.k_test
    sigmoid_temper = tta_cfg.temper

    labels = dataloader.dataset.img2txt
    recall_types = find_recall_types(labels, sims_matrix_i2t, k_test)
    text_ids.to(model.device)
    text_atts.to(model.device)

    score_matrix_i2t = torch.full(
        (len(dataloader.dataset.image), len(dataloader.dataset.text)), -100.0
    ).to(model.device)
    logging_list = []

    start_time = time.time()
    logging.info("    start i2t itm eval... ")
    model.eval()
    with torch.no_grad():
        with tqdm( desc="ITM EVAL", total=sims_matrix_i2t.size(0) ) as tbar:
            for i, sims_i2t in enumerate(sims_matrix_i2t): # 遍历每个image与25010个text的sim_matrix
                topk_sim_i2t, topk_idx_i2t = sims_i2t.topk(k=k_test, dim=0) #sims.shape=25010 topk_sim.shape=128 topk_idx=top128_idx
                image_inputs = vit_feats[i].repeat(k_test, 1, 1).to(model.device) # vit_feats[i].shape=1,677,1408 image_inputs.shape=128,677,1408
                logits = model.compute_itm_logits(
                    image_inputs=image_inputs, #128,677,1408
                    text_ids=text_ids[topk_idx_i2t].to(model.device), #128,35
                    text_atts=text_atts[topk_idx_i2t].to(model.device), #128,35
                ).float() # score.shape=128
                score = logits[:, 1]
                ## score = itm_score + cos_sim
                score_matrix_i2t[i, topk_idx_i2t] = score + topk_sim_i2t.to(model.device)

                ## entropy
                entropy_logits_softmax = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(1).mean()
                entropy_sigmoid_sum = -(F.sigmoid(score) * torch.log(F.sigmoid(score))).sum().mean() / sigmoid_temper
                entropy_sigmoid_mean = -(F.sigmoid(score) * torch.log(F.sigmoid(score))).mean().mean() / sigmoid_temper
                entropy_softmax = -(F.softmax(score, dim=0) * F.log_softmax(score, dim=0)).sum().mean()
                ## logging
                recall_type = recall_types[i]
                logging_list.append({
                    "recall_type" : recall_type,
                    "label" : labels[i],
                    "topk_sim" : topk_sim_i2t.detach().cpu().numpy().tolist(),
                    "topk_idx" : topk_idx_i2t.detach().cpu().numpy().tolist(),
                    "score" : score.detach().cpu().numpy().tolist(),
                    "entropy_logits_softmax" : entropy_logits_softmax.detach().cpu().numpy().tolist(),
                    "entropy_sigmoid_sum" : entropy_sigmoid_sum.detach().cpu().numpy().tolist(),
                    "entropy_sigmoid_mean" : entropy_sigmoid_mean.detach().cpu().numpy().tolist(),
                    "entropy_softmax" : entropy_softmax.detach().cpu().numpy().tolist(),
                })
                ## tqdm logging
                tbar.set_postfix(
                    entropy_logits_softmax=entropy_logits_softmax.detach().cpu().numpy(),
                    entropy_sigmoid_sum=entropy_sigmoid_sum.detach().cpu().numpy(),
                    entropy_sigmoid_mean=entropy_sigmoid_mean.detach().cpu().numpy(),
                    entropy_softmax=entropy_softmax.detach().cpu().numpy())
                tbar.update(1)

    ## save logging json
    logging_list_json = json.dumps(logging_list)
    json_path = os.path.join(registry.get_path("output_dir"), f"eval_epoch{epoch}l_logging_list.json")
    with open(json_path, "w") as f:
        json.dump(logging_list_json, f)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("    i2t itm eval time: {}".format(total_time_str))
    logging.info("compute_i2t_itm_score: end")
    return score_matrix_i2t.cpu().detach().numpy()

def adapt_i2t_itm_score(model, dataloader, task_cfg, optimizer, tta_cfg, sims_matrix_i2t, vit_feats, text_ids, text_atts, epoch=-1):
    logging.info("adapt_i2t_itm_score: start")

    k_test = task_cfg.k_test
    itm_loss_backward_accum_bs = tta_cfg.grad_accum_bs
    sigmoid_temper = tta_cfg.temper

    labels = dataloader.dataset.img2txt
    recall_types = find_recall_types(labels, sims_matrix_i2t, k_test)
    top1_match_coeffis = compute_top1_match_coeffis(sims_matrix_i2t, sims_matrix_i2t.t(), tta_cfg.top1_match_coeffi_src, k_test)

    score_matrix_i2t = torch.full(
        (len(dataloader.dataset.image), len(dataloader.dataset.text)), -100.0
    ).to(model.device)
    iters_entropy_accum = 0.0
    epoch_entropy_list = []
    iters_loss_accum = 0.0
    epoch_loss_list = []
    logging_list = []
    grad_accum_num = 0

    start_time = time.time()
    logging.info("    start i2t itm adapt online Evaluation... ")
    model.eval()
    with torch.enable_grad():
        with tqdm( total=sims_matrix_i2t.size(0) ) as tbar:
            for i, sims_i2t in enumerate(sims_matrix_i2t): # 遍历每个image与25010个text的sim_matrix
                topk_sim_i2t, topk_idx_i2t = sims_i2t.topk(k=k_test, dim=0) #sims.shape=25010 topk_sim.shape=128 topk_idx=top128_idx
                image_inputs = vit_feats[i].repeat(k_test, 1, 1).to(model.device) # vit_feats[i].shape=1,677,1408 image_inputs.shape=128,677,1408
                logits = model.compute_itm_logits(
                    image_inputs=image_inputs, #128,677,1408
                    text_ids=text_ids[topk_idx_i2t].to(model.device), #128,35
                    text_atts=text_atts[topk_idx_i2t].to(model.device), #128,35
                ).float() # score.shape=128
                score = logits[:, 1]

                ## score = itm_score + cos_sim
                score_matrix_i2t[i, topk_idx_i2t] = score + topk_sim_i2t.to(model.device)

                # coeffi
                if tta_cfg.top1_match_coeffi == True:
                    top1_match_coeffi = top1_match_coeffis[i].to(model.device)
                else:
                    top1_match_coeffi = torch.ones(1).to(model.device)
                # itm entropy adapt
                if tta_cfg.entropy_type == "logits_softmax":
                    entropy = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(1)
                elif tta_cfg.entropy_type == "sigmoid_sum":
                    entropy = -(F.sigmoid(score) * torch.log(F.sigmoid(score))).sum() / sigmoid_temper
                elif tta_cfg.entropy_type == "sigmoid_mean":
                    entropy = -(F.sigmoid(score) * torch.log(F.sigmoid(score))).mean() / sigmoid_temper
                elif tta_cfg.entropy_type == "softmax":
                    entropy = -(F.softmax(score, dim=0) * F.log_softmax(score, dim=0)).sum()
                else:
                    entropy = -(F.softmax(score, dim=0) * F.log_softmax(score, dim=0)).sum()
                entropy = entropy.mean()
                loss = entropy / top1_match_coeffi
                loss = loss / itm_loss_backward_accum_bs
                loss.backward()
                grad_accum_num += 1
                if grad_accum_num >= itm_loss_backward_accum_bs or i+1>sims_matrix_i2t.size(0):
                    optimizer.step()
                    optimizer.zero_grad()
                    grad_accum_num = 0

                ## logging json
                recall_type = recall_types[i]
                logging_list.append({
                    "label" : labels[i],
                    "score" : score.detach().cpu().numpy().tolist(),
                    "recall_type" : recall_type,
                    "entropy" : entropy.detach().cpu().numpy().tolist(),
                    "topk_sim_i2t" : topk_sim_i2t.detach().cpu().numpy().tolist(),
                    "topk_idx_i2t" : topk_idx_i2t.detach().cpu().numpy().tolist(),
                    "coeffi": top1_match_coeffi.detach().cpu().numpy().tolist(),
                    "loss" : [  loss * itm_loss_backward_accum_bs for loss in loss.detach().cpu().numpy().tolist()],
                })
                ## log_iters logging
                iters_entropy_accum += entropy.detach().cpu().numpy()
                iters_loss_accum += loss.detach().cpu().numpy() * itm_loss_backward_accum_bs
                if (i+1) % tta_cfg.log_iters == 0 or i+1>sims_matrix_i2t.size(0):
                    logging.info(" ")
                    logging.info(f"[i2t online Evaluation itm adapt] Iteration: {i}, Iters Average Entropy: {iters_entropy_accum / tta_cfg.log_iters}, Iters Average Loss: {iters_loss_accum / tta_cfg.log_iters} ")
                    iters_entropy_accum = 0.0
                    iters_loss_accum = 0.0
                    epoch_entropy_list.append(entropy.detach().cpu().numpy())
                    epoch_loss_list.append(loss.detach().cpu().numpy() * itm_loss_backward_accum_bs)
                ## tqdm logging
                tbar.set_postfix(
                    entropy=entropy.detach().cpu().numpy(),
                    loss=loss.detach().cpu().numpy()*itm_loss_backward_accum_bs,
                    lr=optimizer.param_groups[0]['lr']
                )
                tbar.update(1)

    ## save epoch logging
    logging_list_json = json.dumps(logging_list)
    json_path = os.path.join(registry.get_path("output_dir"), f"epoch{epoch}_logging_list.json")
    with open(json_path, "w") as f:
        json.dump(logging_list_json, f)

    plt.figure()
    plt.plot(epoch_entropy_list)
    plt.savefig(os.path.join(registry.get_path("output_dir"), f"epoch{epoch}_entropy.jpg"))
    plt.figure()
    plt.plot(epoch_loss_list)
    plt.savefig(os.path.join(registry.get_path("output_dir"), f"epoch{epoch}_loss.jpg"))
    # if tta_cfg.debug_visual == True:
    #     plt.show()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("i2t itm adapt online Evaluation time: {}".format(total_time_str))
    logging.info("adapt_i2t_itm_score: end")
    return score_matrix_i2t.cpu().detach().numpy()

# adapt_i2t_itc_sim

# compute_t2i_itm_score

# adapt_t2i_itm_score

# adapt_t2i_itc_sim


def sample_neg_idxs(sims_matrix_i2t, k_tta, k_test, neg_sample_range=[32, 128]):
    """
    Sample negative indices from the similarity matrix.
    Args:
        sims_matrix_i2t: Similarity matrix of shape (num_images, num_texts).
        k_tta: Number of negative samples to sample.
        k_test: Number of top-k samples to consider.
    Returns:
        neg_sims: Negative similarities of shape (num_images, k_tta-1).
        neg_idxs: Indices of the negative samples.
    """
    num_images, num_texts = sims_matrix_i2t.shape
    neg_idxs = []
    neg_sims = []
    for i in range(num_images):
        topk_sim_i2t, topk_idx_i2t = sims_matrix_i2t[i].topk(k=k_test, dim=0)
        # random_idx = torch.randperm(k_test-2*k_tta)[:k_tta-1] + 2*k_tta # 随机采样k_tta-1个负样本索引
        random_idx = torch.randperm(neg_sample_range[1] - neg_sample_range[0])[:k_tta-1] + neg_sample_range[0]
        neg_sims.append(topk_sim_i2t[random_idx])
        neg_idxs.append(topk_idx_i2t[random_idx])
    return torch.stack(neg_sims, dim=0), torch.stack(neg_idxs, dim=0)

def compute_tta_coeffis(sims_matrix_i2t, sims_matrix_t2i, k_test, i2t_temper=100, t2i_temper=20):
    coeffis = []
    for i, sims_i2t in enumerate(sims_matrix_i2t):
        topk_sim_i2t, topk_idx_i2t = sims_i2t.topk(k=k_test, dim=0)

        coeffi = torch.ones(1)
        proba_top1_sim_i2t = F.softmax(topk_sim_i2t * i2t_temper, dim=0)[0] # i2t和t2i的logtis rank/distribution呈现长尾or平均的现象
        proba_sim_t2i_top1_idx_i2t = torch.zeros(1)
        topk_sim_t2i_top1_idx_i2t, topk_idx_t2i_top1_idx_i2t = sims_matrix_t2i[topk_idx_i2t[0]].topk(k=k_test, dim=0)
        if i in topk_idx_t2i_top1_idx_i2t:
            idx_i_in_topk_idx_t2i_top1_idx_i2t = torch.where(topk_idx_t2i_top1_idx_i2t == i)
            proba_sim_t2i_top1_idx_i2t = F.softmax(topk_sim_t2i_top1_idx_i2t * t2i_temper, dim=0)[idx_i_in_topk_idx_t2i_top1_idx_i2t]
        coeffi = torch.exp( 1 - (proba_top1_sim_i2t + proba_sim_t2i_top1_idx_i2t) / 2 )
        coeffis.append(coeffi)
    return coeffis

def find_inter_top1_sample_selection(sims_matrix_i2t, sims_matrix_t2i):
    ss_idxs = []
    for i, sims_i2t in enumerate(sims_matrix_i2t):
        top1_sim_i2t, top1_idx_i2t = sims_i2t.topk(k=1, dim=0) # 获取i2t top1的相似度(top1_sim_i2t)和索引(top1_idx_i2t)
        top1_sim_t2i, top1_idx_t2i  = sims_matrix_t2i[top1_idx_i2t][0].topk(k=1, dim=0) # 获取top1_idx_i2t对应的t2i相似度和索引
        if top1_idx_t2i == i: # 只保留i2t和t2i互为top1的样本
            ss_idxs.append(i)
        # else:
        #     logging.info(f"i2t and t2i not inter top1 sample: i2t_idx={i}, t2i_idx={top1_idx_t2i}, top1_sim_i2t={top1_sim_i2t}, top1_sim_t2i={top1_sim_t2i}")
    return ss_idxs #4286个互为top1的样本

## top1 sample selection + 负样本采样计算softmax_entropy
def adapt_i2t_itm_score_v2(model, dataloader, task_cfg, optimizer, tta_cfg, sims_matrix_i2t, vit_feats, text_ids, text_atts, epoch=-1):
    logging.info("adapt_i2t_itm_score_v2: start")
    ## return output
    score_matrix_i2t = torch.full(
        (len(dataloader.dataset.image), len(dataloader.dataset.text)), -100.0
    ).to(model.device)
    scores_mat = torch.full(
        (len(dataloader.dataset.image), tta_cfg.k_tta), -100.0
    ).to(model.device)
    ## nn.distributed
    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_i2t.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_i2t.size(0), start + step)
    sims_matrix_i2t = sims_matrix_i2t[start:end] # 只取当前rank的样本
    vit_feats = vit_feats[start:end] # 只取当前rank的样本
    # text_ids = text_ids[start:end] # 只取当前rank的样本
    # text_atts = text_atts[start:end] # 只取当前rank的样本
    score_matrix_i2t_ = score_matrix_i2t[start:end].cpu() # 只取当前rank的样本
    scores_mat_ = scores_mat[start:end].cpu()
    ## sampling stretegy
    k_test = task_cfg.k_test
    labels = dataloader.dataset.img2txt
    ## 获取metric标签用于可视化
    recall_types = find_recall_types(labels, sims_matrix_i2t, k_test=10)
    ## 采样正样本
    # pos_sample_range =  tta_cfg.pos_sample_range if hasattr(tta_cfg, "pos_sample_range") else [0, 1] # 正样本采样范围
    top1_sims, top1_idxs = sims_matrix_i2t.topk(k=1, dim=1) #正样本直接使用top1 TODO 使用top5采样一个正样本,但是这里相似度top5不一定就是5个label，所以还需要确定？
    top1_sims, top1_idxs =top1_sims[:,0], top1_idxs[:,0]
    ## 采样困难负样本
    neg_sample_range = tta_cfg.neg_sample_range if hasattr(tta_cfg, "neg_sample_range") else [32, 128] # 负样本采样范围
    neg_sims, neg_idxs = sample_neg_idxs(sims_matrix_i2t, tta_cfg.k_tta, k_test, neg_sample_range) # 负样本采样k_tta-1个, 根据score数值可视化差异确定采样范围
    ## 拼接正负样本
    sampled_sims_matrix_i2t = []
    sampled_sims_idx_i2t = []
    for i in range(sims_matrix_i2t.size(0)):
        sampled_sim = torch.concatenate((sims_matrix_i2t[i,top1_idxs[i]].reshape(-1), sims_matrix_i2t[i,neg_idxs[i]]))
        sampled_idx = torch.concatenate((top1_idxs[i].reshape(-1), neg_idxs[i]))
        sampled_sims_matrix_i2t.append(sampled_sim)
        sampled_sims_idx_i2t.append(sampled_idx)
    sampled_sims_matrix_i2t = torch.stack(sampled_sims_matrix_i2t) # shape=(5000, k_tta)
    sampled_sims_idx_i2t = torch.stack(sampled_sims_idx_i2t) # shape=(5000, k_tta)
    ## entropy coeffi
    if tta_cfg.top1_match_coeffi == True:
        tta_coeffis = compute_tta_coeffis(sims_matrix_i2t, sims_matrix_i2t.t(), k_test, tta_cfg.coeffi_i2t_temper, tta_cfg.coeffi_t2i_temper)
    else:
        tta_coeffis = [ torch.ones(1) for _ in range(sims_matrix_i2t.size(0)) ]
    ## score temperature
    score_temper_ = tta_cfg.score_temper if hasattr(tta_cfg, "score_temper") else 1.0

    ## sample selection stretegy
    if tta_cfg.sample_selection == "top1":
        ss_idxs = find_inter_top1_sample_selection(sims_matrix_i2t, sims_matrix_i2t.t()) # 找到i2t和t2i互为top1的样本索引
        # 只保留top1 sample selection的样本
        # sims_matrix_i2t = sims_matrix_i2t[ss_idxs]
        vit_feats = vit_feats[ss_idxs]
        # text_ids = text_ids[ss_idxs]
        # text_atts = text_atts[ss_idxs]
        labels = [labels[i] for i in ss_idxs]
        recall_types = [recall_types[i] for i in ss_idxs]
        # top1_sims, top1_idxs = top1_sims[ss_idxs], top1_idxs[ss_idxs]
        # neg_sims, neg_idxs = neg_sims[ss_idxs], neg_idxs[ss_idxs]
        sampled_sims_matrix_i2t = sampled_sims_matrix_i2t[ss_idxs]
        tta_coeffis = [tta_coeffis[i] for i in ss_idxs]
        score_matrix_i2t_2_ = score_matrix_i2t_[ss_idxs]
        scores_mat_2_ = scores_mat_[ss_idxs]
    else:
        ss_idxs = torch.arange(0, sims_matrix_i2t.size(0)) # 全部样本
        score_matrix_i2t_2_ = score_matrix_i2t_[ss_idxs]
        scores_mat_2_ = scores_mat_[ss_idxs]
    logging.info("    number of sample_selection: {}".format(len(ss_idxs)))

    ## training loop
    iters_entropy_accum = 0.0
    epoch_entropy_list = []
    iters_loss_accum = 0.0
    epoch_loss_list = []
    logging_list = []
    grad_accum_num = 0

    start_time = time.time()
    logging.info("    start i2t itm adapt... ")
    model.eval()
    with torch.enable_grad():
        with tqdm(desc=f"ITM EVAL rank{rank}", total = sampled_sims_matrix_i2t.size(0) // tta_cfg.tta_bs + 1 ) as tbar:
            for iter, idx in enumerate(range(0, sampled_sims_matrix_i2t.size(0), tta_cfg.tta_bs)): # 遍历每个image与25010个text的sim_matrix
                idx_end = min(idx+tta_cfg.tta_bs, sampled_sims_matrix_i2t.size(0))
                sims_i2t = sampled_sims_matrix_i2t[idx:idx_end] # 取出当前batch的sims_i2t
                idxs_i2t = sampled_sims_idx_i2t[idx:idx_end] # 取出当前batch的idxs_i2t
                # 将shape=(bs, k_tta)的向量idxs_i2t展开成shape=(bs*k_tta)维度，且向量中每个元素顺序不变
                image_inputs = vit_feats[idx:idx_end].unsqueeze(1).repeat(1, tta_cfg.k_tta, 1, 1) # vit_feats[i].shape=bs,677,1408 image_inputs.shape=bs,k_tta,677,1408
                image_inputs = image_inputs.reshape(-1, image_inputs.size(2), image_inputs.size(3)) # bs*k_tta,677,1408
                text_ids_inputs = text_ids[idxs_i2t]
                text_ids_inputs = text_ids_inputs.reshape(-1, text_ids_inputs.size(-1)) # bs*k_tta,35
                text_atts_inputs = text_atts[idxs_i2t]
                text_atts_inputs = text_atts_inputs.reshape(-1, text_atts_inputs.size(-1)) # bs*k_tta,35
                logits = model.compute_itm_logits(
                    image_inputs=image_inputs.to(model.device), #bs*k_tta,677,1408
                    text_ids=text_ids_inputs.to(model.device), #bs*k_tta,35
                    text_atts=text_atts_inputs.to(model.device), #bs*k_tta,35
                    # TODONE 加一个attention_mask，让每个image_inputs仅和对应的text_ids做cross-attention，节约显存和算力
                    #【无需这样做，构造的bs * tta_bs样本对已经符合image仅和对应的text计算cross-attention】
                ).float() # logits.shape=bs*k_tta, 2
                logits = logits.reshape(-1, tta_cfg.k_tta, 2) # logits.shape=bs, k_tta, 2
                score = logits[..., 1] # score.shape=bs, k_tta
                # score。show，观察pos和neg样本的差异，即可以区分又不至于差异太大

                ## score = itm_score + cos_sim
                for i, bs_idx in enumerate(range(idx,idx_end)):
                    score_matrix_i2t_2_[bs_idx, idxs_i2t[i]] = score[i].detach().cpu() + sims_i2t.reshape(-1, tta_cfg.k_tta)[i].detach().cpu()
                    scores_mat_2_[bs_idx, ...] = score[i].detach().cpu()

                # coeffi
                tta_coeffi = torch.Tensor(tta_coeffis[idx:idx_end]).to(model.device)
                # temperature
                score_temper = torch.Tensor([score_temper_]).to(model.device)
                # itm entropy adapt
                entropy = -(F.softmax(score * score_temper, dim=-1) * F.log_softmax(score * score_temper, dim=-1)).sum(-1) # score * temper # loss = pos-neg softmax entropy
                # entropy = (-(F.sigmoid(score * score_temper) * F.log(F.sigmoid(score * score_temper)))).sum(-1) # loss = probability sigmoid entropy
                loss = entropy / tta_coeffi
                loss = loss.mean()
                loss = loss / tta_cfg.grad_accum_bs
                loss.backward()
                grad_accum_num += 1
                if grad_accum_num >= tta_cfg.grad_accum_bs or iter + 1 > sampled_sims_matrix_i2t.size(0)//tta_cfg.tta_bs + 1 :
                    optimizer.step()
                    optimizer.zero_grad()
                    grad_accum_num = 0

                ## logging json
                logging_list.append({
                    "sample_idx" : [i for i in range(idx,idx_end)],
                    "label" : [labels[i] for i in range(idx,idx_end)],
                    "score" : score.detach().cpu().numpy().tolist(),
                    "recall_type" : recall_types[idx:idx_end],
                    "sims_i2t" : sims_i2t.reshape(-1, tta_cfg.k_tta).detach().cpu().numpy().tolist(),
                    "idxs_i2t" : idxs_i2t.reshape(-1, tta_cfg.k_tta).detach().cpu().numpy().tolist(),
                    "tta_coeffi": tta_coeffi.detach().cpu().numpy().tolist(),
                    "entropy" : entropy.detach().cpu().numpy().tolist(),
                    "loss" : loss.detach().cpu().numpy()  * tta_cfg.grad_accum_bs,
                })
                ## log_iters logging
                epoch_entropy_list.append(entropy.mean().detach().cpu().numpy())
                epoch_loss_list.append(loss.detach().cpu().numpy() * tta_cfg.grad_accum_bs)
                iters_entropy_accum += entropy.mean().detach().cpu().numpy()
                iters_loss_accum += loss.detach().cpu().numpy() * tta_cfg.grad_accum_bs
                if (iter+1) % tta_cfg.log_iters == 0 or iter+1>sims_matrix_i2t.size(0):
                    logging.info(" ")
                    logging.info(f"[ITM ADAPT rank{rank}] Iteration: {iter}, Iters Average Entropy: {iters_entropy_accum / tta_cfg.log_iters}, Iters Average Loss: {iters_loss_accum / tta_cfg.log_iters}, Learning Rate: {optimizer.param_groups[0]['lr']}")
                    iters_entropy_accum = 0.0
                    iters_loss_accum = 0.0
                # ## tqdm logging
                # tbar.set_postfix(
                #     entropy=entropy.mean().detach().cpu().numpy(),
                #     loss=loss.detach().cpu().numpy()*tta_cfg.grad_accum_bs,
                #     lr=optimizer.param_groups[0]['lr']
                # )
                # tbar.update(1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("    i2t itm adapt time: {}".format(total_time_str))

    ## torch.distributed
    score_matrix_i2t_[ss_idxs] = score_matrix_i2t_2_
    score_matrix_i2t[start:end] = score_matrix_i2t_.to(model.device)# 将当前rank的score_matrix_i2t放回到全局的score_matrix_i2t中
    scores_mat_[ss_idxs] = scores_mat_2_
    scores_mat[start:end] = scores_mat_.to(model.device)
    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            scores_mat, op=torch.distributed.ReduceOp.SUM
        )

    ## save epoch logging
    logging_list_json = json.dumps(logging_list)
    json_path = os.path.join(registry.get_path("output_dir"), f"result/rank{rank}/tta_epoch{epoch}_logging_list.json")
    with open(json_path, "w") as f:
        json.dump(logging_list_json, f)
    ## plt entropy
    plt.figure()
    plt.plot(epoch_entropy_list)
    plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{rank}/tta_epoch{epoch}_entropy.jpg"))
    ## plt loss
    plt.figure()
    plt.plot(epoch_loss_list)
    plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{rank}/tta_epoch{epoch}_loss.jpg"))
    ## plt score
    if is_main_process():
        plt.figure(figsize=(32,8))
        plt.plot(scores_mat[:,0].cpu().detach().numpy(), alpha=0.7)
        plt.plot(scores_mat[:,1:].cpu().detach().numpy().mean(axis=1), alpha=0.7)
        plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/tta_epoch{epoch}_score_distribution.jpg"))
    # if tta_cfg.debug_visual == True:
    #     plt.show()

    logging.info("adapt_i2t_itm_score_v2: end")
    return score_matrix_i2t.cpu().detach().numpy(), scores_mat.cpu().detach().numpy()


def compute_i2t_itm_score_v2(model, dataloader, task_cfg, tta_cfg, sims_matrix_i2t, vit_feats, text_ids, text_atts, epoch=-1):
    logging.info("compute_i2t_itm_score: start")

    k_test = task_cfg.k_test

    labels = dataloader.dataset.img2txt
    recall_types = find_recall_types(labels, sims_matrix_i2t, k_test)
    text_ids.to(model.device)
    text_atts.to(model.device)
    logging.info("    number of dataloader: {}".format(len(dataloader)))

    score_matrix_i2t = torch.full(
        (len(dataloader.dataset.image), len(dataloader.dataset.text)), -100.0
    ).to(model.device)
    scores_mat = torch.full(
        (len(dataloader.dataset.image), k_test), -100.0
    ).to(model.device)
    logging_list = []

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_i2t.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_i2t.size(0), start + step)

    start_time = time.time()
    logging.info("    start i2t itm eval... ")
    model.eval()
    with torch.no_grad():
        # with tqdm( desc=f"ITM EVAL rank{rank}", total=sims_matrix_i2t[start:end].size(0) ) as tbar:
        if 1:
            for i, sims_i2t in enumerate(sims_matrix_i2t[start:end]): # 遍历每个image与25010个text的sim_matrix
                topk_sim_i2t, topk_idx_i2t = sims_i2t.topk(k=k_test, dim=0) #sims.shape=25010 topk_sim.shape=128 topk_idx=top128_idx
                image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device) # vit_feats[i].shape=1,677,1408 image_inputs.shape=128,677,1408
                logits = model.compute_itm_logits(
                    image_inputs=image_inputs, #128,677,1408
                    text_ids=text_ids[topk_idx_i2t].to(model.device), #128,35
                    text_atts=text_atts[topk_idx_i2t].to(model.device), #128,35
                ).float() # score.shape=128
                score = logits[:, 1]
                ## score = itm_score + cos_sim
                score_matrix_i2t[start + i, topk_idx_i2t] = score + topk_sim_i2t.to(model.device)
                scores_mat[start + i, ...] = score

                ## entropy
                entropy = -(F.softmax(score, dim=-1) * F.log_softmax(score, dim=-1)).sum(-1).mean()
                ## logging
                recall_type = recall_types[i]
                logging_list.append({
                    "recall_type" : recall_type,
                    "label" : labels[i],
                    "topk_sim" : topk_sim_i2t.detach().cpu().numpy().tolist(),
                    "topk_idx" : topk_idx_i2t.detach().cpu().numpy().tolist(),
                    "score" : score.detach().cpu().numpy().tolist(),
                    "entropy" : entropy.detach().cpu().numpy().tolist(),
                })
                ## log_iters logging
                if (i+1) % tta_cfg.log_iters == 0 or i+1>sims_matrix_i2t[start:end].size(0):
                    logging.info(f"[ITM ADAPT rank{rank}] Iteration: {i}, Recall Type: {recall_type}, Iteration Entropy: {entropy.detach().cpu().numpy()}")
                # ## tqdm logging
                # tbar.set_postfix(recall_type=recall_type, entropy=entropy.detach().cpu().numpy())
                # tbar.update(1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("    i2t itm eval time: {}".format(total_time_str))

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            scores_mat, op=torch.distributed.ReduceOp.SUM
        )

    # ## save logging json
    # logging_list_json = json.dumps(logging_list)
    # json_path = os.path.join(registry.get_path("output_dir"), f"result/rank{rank}/eval_epoch{epoch}_logging_list.json")
    # with open(json_path, "w") as f:
    #     json.dump(logging_list_json, f)
    # ## plt score
    # if is_main_process():
    #     plt.figure(figsize=(32,8))
    #     plt.plot(scores_mat[:,0].cpu().detach().numpy(), alpha=0.7)
    #     plt.plot(scores_mat[:,0:5].cpu().detach().numpy().mean(axis=1), alpha=0.7)
    #     plt.plot(scores_mat[:,5:5+tta_cfg.k_tta-1].cpu().detach().numpy().mean(axis=1), alpha=0.7)
    #     plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/eval_epoch{epoch}_score_distribution.jpg"))

    logging.info("compute_i2t_itm_score: end")
    return score_matrix_i2t.cpu().detach().numpy(), scores_mat.cpu().detach().numpy(), logging_list

## top1 sample selection + 负样本采样计算softmax_entropy
def adapt_t2i_itm_score_v2(model, dataloader, task_cfg, optimizer, tta_cfg, sims_matrix_t2i, vit_feats, text_ids, text_atts, epoch=-1):
    logging.info("adapt_t2i_itm_score_v2: start")
    ## return output
    score_matrix_t2i = torch.full(
        (len(dataloader.dataset.text), len(dataloader.dataset.image)), -100.0
    ).to(model.device)
    scores_mat = torch.full(
        (len(dataloader.dataset.text), tta_cfg.k_tta), -100.0
    ).to(model.device)
    ## nn.distributed
    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_t2i.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_t2i.size(0), start + step)
    sims_matrix_t2i = sims_matrix_t2i[start:end] # 只取当前rank的样本
    # vit_feats = vit_feats[start:end] # 只取当前rank的样本
    text_ids = text_ids[start:end] # 只取当前rank的样本
    text_atts = text_atts[start:end] # 只取当前rank的样本
    score_matrix_t2i_ = score_matrix_t2i[start:end].cpu() # 只取当前rank的样本
    scores_mat_ = scores_mat[start:end].cpu()
    ## sampling stretegy
    k_test = task_cfg.k_test
    labels = dataloader.dataset.txt2img
    recall_types = find_recall_types(labels, sims_matrix_t2i, k_test=10)
    ## 采样正样本
    top1_sims, top1_idxs = sims_matrix_t2i.topk(k=1, dim=1) #正样本直接使用top1; t2i仅有一个label所以直接取top1作为正样本即可
    top1_sims, top1_idxs =top1_sims[:,0], top1_idxs[:,0]
    ## 采样困难负样本
    neg_sample_range = tta_cfg.neg_sample_range if hasattr(tta_cfg, "neg_sample_range") else [32, 128] # 负样本采样范围
    neg_sims, neg_idxs = sample_neg_idxs(sims_matrix_t2i, tta_cfg.k_tta, k_test, neg_sample_range) # 负样本采样k_tta-1个,根据score数值差异确定采样范围
    ## 拼接正负样本
    sampled_sims_matrix_t2i = []
    sampled_sims_idx_t2i = []
    for i in range(sims_matrix_t2i.size(0)):
        sampled_sim = torch.concatenate((sims_matrix_t2i[i,top1_idxs[i]].reshape(-1), sims_matrix_t2i[i,neg_idxs[i]]))
        sampled_idx = torch.concatenate((top1_idxs[i].reshape(-1), neg_idxs[i]))
        sampled_sims_matrix_t2i.append(sampled_sim)
        sampled_sims_idx_t2i.append(sampled_idx)
    sampled_sims_matrix_t2i = torch.stack(sampled_sims_matrix_t2i) # shape=(5000, k_tta)
    sampled_sims_idx_t2i = torch.stack(sampled_sims_idx_t2i) # shape=(5000, k_tta)
    ## entropy coeffi
    if tta_cfg.top1_match_coeffi == True:
        tta_coeffis = compute_tta_coeffis(sims_matrix_t2i, sims_matrix_t2i.t(), k_test, tta_cfg.coeffi_i2t_temper, tta_cfg.coeffi_t2i_temper)
    else:
        tta_coeffis = [ torch.ones(1) for _ in range(sims_matrix_t2i.size(0)) ]
    ## score temperature
    score_temper_ = tta_cfg.score_temper if hasattr(tta_cfg, "score_temper") else 1.0

    ## sample selection stretegy
    if tta_cfg.sample_selection == "top1":
        ss_idxs = find_inter_top1_sample_selection(sims_matrix_t2i, sims_matrix_t2i.t()) # 找到i2t和t2i互为top1的样本索引
        # 只保留top1 sample selection的样本
        # sims_matrix_t2i = sims_matrix_t2i[ss_idxs]
        # vit_feats = vit_feats[ss_idxs]
        text_ids = text_ids[ss_idxs]
        text_atts = text_atts[ss_idxs]
        labels = [labels[i] for i in ss_idxs]
        recall_types = [recall_types[i] for i in ss_idxs]
        # top1_sims, top1_idxs = top1_sims[ss_idxs], top1_idxs[ss_idxs]
        # neg_sims, neg_idxs = neg_sims[ss_idxs], neg_idxs[ss_idxs]
        sampled_sims_matrix_t2i = sampled_sims_matrix_t2i[ss_idxs]
        tta_coeffis = [tta_coeffis[i] for i in ss_idxs]
        score_matrix_t2i_2_ = score_matrix_t2i_[ss_idxs]
        scores_mat_2_ = scores_mat_[ss_idxs]
    else:
        ss_idxs = torch.arange(0, sims_matrix_t2i.size(0)) # 全部样本
        score_matrix_i2t_2_ = score_matrix_i2t_[ss_idxs]
        scores_mat_2_ = scores_mat_[ss_idxs]
    logging.info("    number of sample_selection: {}".format(len(ss_idxs)))

    ## training loop
    iters_entropy_accum = 0.0
    epoch_entropy_list = []
    iters_loss_accum = 0.0
    epoch_loss_list = []
    logging_list = []
    grad_accum_num = 0

    start_time = time.time()
    logging.info("    start t2i itm adapt... ")
    model.eval()
    with torch.enable_grad():
        with tqdm( desc=f"ITM EVAL rank{rank}", total = sampled_sims_matrix_t2i.size(0) // tta_cfg.tta_bs + 1 ) as tbar:
            for iter, idx in enumerate(range(0, sampled_sims_matrix_t2i.size(0), tta_cfg.tta_bs)): # 遍历每个image与25010个text的sim_matrix
                idx_end = min(idx+tta_cfg.tta_bs, sampled_sims_matrix_t2i.size(0))
                sims_t2i = sampled_sims_matrix_t2i[idx:idx_end] # 取出当前batch的sims_t2i
                idxs_t2i = sampled_sims_idx_t2i[idx:idx_end] # 取出当前batch的idxs_t2i
                # 将shape=(bs, k_tta)的向量idxs_t2i展开成shape=(bs*k_tta)维度，且向量中每个元素顺序不变
                image_inputs = vit_feats[idxs_t2i] # vit_feats[i].shape=bs,677,1408 image_inputs.shape=bs,k_tta,677,1408
                image_inputs = image_inputs.reshape(-1, image_inputs.size(2), image_inputs.size(3)) # bs*k_tta,677,1408
                text_ids_inputs = text_ids[idx:idx_end]
                text_ids_inputs = text_ids_inputs.unsqueeze(1).repeat(1, tta_cfg.k_tta, 1).reshape(-1, text_ids_inputs.size(-1)) # bs*k_tta,35
                text_atts_inputs = text_atts[idx:idx_end]
                text_atts_inputs = text_atts_inputs.unsqueeze(1).repeat(1, tta_cfg.k_tta, 1).reshape(-1, text_atts_inputs.size(-1)) # bs*k_tta,35
                logits = model.compute_itm_logits(
                    image_inputs=image_inputs.to(model.device), #bs*k_tta,677,1408
                    text_ids=text_ids_inputs.to(model.device), #bs*k_tta,35
                    text_atts=text_atts_inputs.to(model.device), #bs*k_tta,35
                    # TODONE 加一个attention_mask，让每个image_inputs仅和对应的text_ids做cross-attention，节约显存和算力
                    #【无需这样做，构造的bs * tta_bs样本对已经符合image仅和对应的text计算cross-attention】
                ).float() # logits.shape=bs*k_tta, 2
                logits = logits.reshape(-1, tta_cfg.k_tta, 2) # logits.shape=bs, k_tta, 2
                score = logits[..., 1] # score.shape=bs, k_tta
                # score。show，观察pos和neg样本的差异，即可以区分又不至于差异太大

                ## score = itm_score + cos_sim
                for i, bs_idx in enumerate(range(idx,idx_end)):
                    score_matrix_t2i_2_[bs_idx, idxs_t2i[i]] = score[i].detach().cpu() + sims_t2i.reshape(-1, tta_cfg.k_tta)[i].detach().cpu()
                    scores_mat_2_[bs_idx, ...] = score[i].detach().cpu()

                # coeffi
                tta_coeffi = torch.Tensor(tta_coeffis[idx:idx_end]).to(model.device)
                # temperature
                score_temper = torch.Tensor([score_temper_]).to(model.device)
                # itm entropy adapt
                entropy = -(F.softmax(score * score_temper, dim=-1) * F.log_softmax(score * score_temper, dim=-1)).sum(-1)# score * temper
                loss = entropy / tta_coeffi
                loss = loss.mean()
                loss = loss / tta_cfg.grad_accum_bs
                loss.backward()
                grad_accum_num += 1
                if grad_accum_num >= tta_cfg.grad_accum_bs or iter + 1 > sampled_sims_matrix_t2i.size(0)//tta_cfg.tta_bs + 1 :
                    optimizer.step()
                    optimizer.zero_grad()
                    grad_accum_num = 0

                ## logging json
                logging_list.append({
                    "sample_idx" : [i for i in range(idx,idx_end)],
                    "label" : [labels[i] for i in range(idx,idx_end)],
                    "score" : score.detach().cpu().numpy().tolist(),
                    "recall_type" : recall_types[idx:idx_end],
                    "sims_t2i" : sims_t2i.reshape(-1, tta_cfg.k_tta).detach().cpu().numpy().tolist(),
                    "idxs_t2i" : idxs_t2i.reshape(-1, tta_cfg.k_tta).detach().cpu().numpy().tolist(),
                    "tta_coeffi": tta_coeffi.detach().cpu().numpy().tolist(),
                    "entropy" : entropy.detach().cpu().numpy().tolist(),
                    "loss" : loss.detach().cpu().numpy()  * tta_cfg.grad_accum_bs,
                })
                ## log_iters logging
                epoch_entropy_list.append(entropy.mean().detach().cpu().numpy())
                epoch_loss_list.append(loss.detach().cpu().numpy() * tta_cfg.grad_accum_bs)
                iters_entropy_accum += entropy.mean().detach().cpu().numpy()
                iters_loss_accum += loss.detach().cpu().numpy() * tta_cfg.grad_accum_bs
                if (iter+1) % tta_cfg.log_iters == 0 or iter+1>sims_matrix_t2i.size(0):
                    logging.info(" ")
                    logging.info(f"[ITM ADAPT rank{rank}] Iteration: {iter}, Iters Average Entropy: {iters_entropy_accum / tta_cfg.log_iters}, Iters Average Loss: {iters_loss_accum / tta_cfg.log_iters}, Learning Rate: {optimizer.param_groups[0]['lr']}")
                    iters_entropy_accum = 0.0
                    iters_loss_accum = 0.0
                # ## tqdm logging
                # tbar.set_postfix(
                #     entropy=entropy.mean().detach().cpu().numpy(),
                #     loss=loss.detach().cpu().numpy()*tta_cfg.grad_accum_bs,
                #     lr=optimizer.param_groups[0]['lr']
                # )
                # tbar.update(1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("    t2i itm adapt time: {}".format(total_time_str))

    ## torch.distributed
    score_matrix_t2i_[ss_idxs] = score_matrix_t2i_2_
    score_matrix_t2i[start:end] = score_matrix_t2i_.to(model.device)# 将当前rank的score_matrix_t2i放回到全局的score_matrix_t2i中
    scores_mat_[ss_idxs] = scores_mat_2_
    scores_mat[start:end] = scores_mat_.to(model.device)
    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            scores_mat, op=torch.distributed.ReduceOp.SUM
        )

    ## save epoch logging
    logging_list_json = json.dumps(logging_list)
    json_path = os.path.join(registry.get_path("output_dir"), f"result/rank{rank}/tta_epoch{epoch}_logging_list.json")
    with open(json_path, "w") as f:
        json.dump(logging_list_json, f)
    ## plt entropy
    plt.figure()
    plt.plot(epoch_entropy_list)
    plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{rank}/tta_epoch{epoch}_entropy.jpg"))
    ## plt loss
    plt.figure()
    plt.plot(epoch_loss_list)
    plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{rank}/tta_epoch{epoch}_loss.jpg"))
    ## plt score
    if is_main_process():
        plt.figure(figsize=(32,8))
        plt.plot(scores_mat[:,0].cpu().detach().numpy(), alpha=0.7)
        plt.plot(scores_mat[:,1:].cpu().detach().numpy().mean(axis=1), alpha=0.7)
        plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/tta_epoch{epoch}_score_distribution.jpg"))
    # if tta_cfg.debug_visual == True:
    #     plt.show()

    logging.info("adapt_t2i_itm_score_v2: end")
    return score_matrix_t2i.cpu().detach().numpy(), scores_mat.cpu().detach().numpy()


def compute_t2i_itm_score_v2(model, dataloader, task_cfg, tta_cfg, sims_matrix_t2i, vit_feats, text_ids, text_atts, epoch=-1):
    logging.info("compute_t2i_itm_score: start")

    k_test = task_cfg.k_test

    labels = dataloader.dataset.txt2img
    recall_types = find_recall_types(labels, sims_matrix_t2i, k_test)
    text_ids.to(model.device)
    text_atts.to(model.device)
    logging.info("    number of dataloader: {}".format(len(dataloader)))

    score_matrix_t2i = torch.full(
        (len(dataloader.dataset.text), len(dataloader.dataset.image)),  -100.0
    ).to(model.device)
    scores_mat = torch.full(
        (len(dataloader.dataset.text), k_test), -100.0
    ).to(model.device)
    logging_list = []

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_t2i.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_t2i.size(0), start + step)

    start_time = time.time()
    logging.info("    start t2i itm eval... ")
    model.eval()
    with torch.no_grad():
        # with tqdm( desc=f"ITM EVAL rank{rank}", total=sims_matrix_t2i[start:end].size(0) ) as tbar:
            for i, sims_t2i in enumerate(sims_matrix_t2i[start:end]): # 遍历每个image与25010个text的sim_matrix
                topk_sim_t2i, topk_idx_t2i = sims_t2i.topk(k=k_test, dim=0) #sims.shape=25010 topk_sim.shape=128 topk_idx=top128_idx
                image_inputs = vit_feats[topk_idx_t2i].to(model.device) # vit_feats[i].shape=1,677,1408 image_inputs.shape=128,677,1408
                logits = model.compute_itm_logits(
                    image_inputs=image_inputs, #128,677,1408
                    text_ids=text_ids[start+i].repeat(k_test, 1).to(model.device), #128,35
                    text_atts=text_atts[start+i].repeat(k_test, 1).to(model.device), #128,35
                ).float() # score.shape=128
                score = logits[:, 1]
                ## score = itm_score + cos_sim
                score_matrix_t2i[start + i, topk_idx_t2i] = score + topk_sim_t2i.to(model.device)
                scores_mat[start + i, :] = score

                ## entropy
                entropy = -(F.softmax(score, dim=-1) * F.log_softmax(score, dim=-1)).sum(-1).mean()
                ## logging
                recall_type = recall_types[i]
                logging_list.append({
                    "recall_type" : recall_type,
                    "label" : labels[i],
                    "topk_sim" : topk_sim_t2i.detach().cpu().numpy().tolist(),
                    "topk_idx" : topk_idx_t2i.detach().cpu().numpy().tolist(),
                    "score" : score.detach().cpu().numpy().tolist(),
                    "entropy" : entropy.detach().cpu().numpy().tolist(),
                })
                ## log_iters logging
                if (i+1) % tta_cfg.log_iters == 0 or i+1>sims_matrix_t2i[start:end].size(0):
                    logging.info(f"[ITM ADAPT] Iteration: {i}, Recall Type: {recall_type}, Iteration Entropy: {entropy.detach().cpu().numpy()}")
                # ## tqdm logging
                # tbar.set_postfix(
                #     recall_type=recall_type,
                #     entropy=entropy.detach().cpu().numpy())
                # tbar.update(1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("    t2i itm eval time: {}".format(total_time_str))

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            scores_mat, op=torch.distributed.ReduceOp.SUM
        )

    # ## save logging json
    # logging_list_json = json.dumps(logging_list)
    # json_path = os.path.join(registry.get_path("output_dir"), f"result/rank{rank}/eval_epoch{epoch}_logging_list.json")
    # with open(json_path, "w") as f:
    #     json.dump(logging_list_json, f)
    # ## plt score
    # if is_main_process():
    #     plt.figure(figsize=(32,8))
    #     plt.plot(scores_mat[:,0].cpu().detach().numpy(), alpha=0.7)
    #     plt.plot(scores_mat[:,1:1+tta_cfg.k_tta-1].cpu().detach().numpy().mean(axis=1), alpha=0.7)
    #     plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/eval_epoch{epoch}_score_distribution.jpg"))

    logging.info("compute_t2i_itm_score: end")
    return score_matrix_t2i.cpu().detach().numpy(), scores_mat.cpu().detach().numpy(), logging_list


def plt_itm_score(scores, task="i2t", mode="tta"):
    if task == "i2t":
        if mode == "tta":
            plt.figure(figsize=(32,8))
            plt.plot(scores[:,0], alpha=0.7)
            plt.plot(scores[:,1:].mean(axis=1), alpha=0.7)
            plt.legend(["Positive top1 Score", "Negative Score top6-8 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_epochs_score_distribution.jpg"))

            # 计算每100个iter的score均值，输出25010/100维向量
            plt.figure(figsize=(32,8))
            avg_scores_pos = scores[:,0].reshape(-1, 100).mean(axis=1)
            avg_scores_neg = scores[:,1:].mean(axis=1).reshape(-1, 100).mean(axis=1)
            plt.plot(avg_scores_pos, marker='o')
            plt.plot(avg_scores_neg, marker='*')
            plt.title("Averaged ITM Score (every 100 samples)")
            plt.xlabel("Batch (100 samples per batch)")
            plt.ylabel("Average Score")
            plt.legend(["Positive top1 Score", "Negative Score top6-8 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_epochs_avg100_score_distribution.jpg"))
        elif mode == "eval":
            # pass ## ipynb script
            plt.figure(figsize=(32,8))
            plt.plot(scores[:,0:5].mean(axis=1), alpha=0.7)
            plt.plot(scores[:,5:8].mean(axis=1), alpha=0.7)
            plt.legend(["Positive Score top1-5 Mean", "Negative Score Top6-8 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_epochs_score_distribution.jpg"))

            # 计算每100个iter的score均值，输出25010/100维向量
            plt.figure(figsize=(32,8))
            avg_scores_pos = scores[:,:5].mean(axis=1).reshape(-1, 100).mean(axis=1)
            avg_scores_neg = scores[:,5:8].mean(axis=1).reshape(-1, 100).mean(axis=1)
            plt.plot(avg_scores_pos, marker='o')
            plt.plot(avg_scores_neg, marker='*')
            plt.title("Averaged ITM Score (every 100 samples)")
            plt.xlabel("Batch (100 samples per batch)")
            plt.ylabel("Average Score")
            plt.legend(["Positive Score top1-5 Mean", "Negative Score Top6-8 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_epochs_avg100_score_distribution.jpg"))
    elif task == "t2i":
        if mode == "tta":
            plt.figure(figsize=(32,8))
            plt.plot(scores[:,0], alpha=0.7)
            plt.plot(scores[:,1:].mean(axis=1), alpha=0.7)
            plt.legend(["Positive top1 Score", "Negative Score top2-4 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_epochs_score_distribution.jpg"))

            # 计算每100个iter的score均值，输出 5000/100维向量
            plt.figure(figsize=(32,8))
            avg_scores_pos = scores[:25000,0].reshape(-1, 100).mean(axis=1)
            avg_scores_neg = scores[:25000,1:].mean(axis=1).reshape(-1, 100).mean(axis=1)
            plt.plot(avg_scores_pos, marker='o')
            plt.plot(avg_scores_neg, marker='*')
            plt.title("Averaged ITM Score (every 100 samples)")
            plt.xlabel("Batch (100 samples per batch)")
            plt.ylabel("Average Score")
            plt.legend(["Positive top1 Score", "Negative Score top2-4 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_epochs_avg100_score_distribution.jpg"))
        elif mode == "eval":
            # pass ## ipynb script
            plt.figure(figsize=(32,8))
            plt.plot(scores[:,0], alpha=0.7)
            plt.plot(scores[:,1:4].mean(axis=1), alpha=0.7)
            plt.legend(["Positive top1 Score", "Negative Score Top2-4 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_epochs_score_distribution.jpg"))

            # 计算每100个iter的score均值，输出 5000/100维向量
            plt.figure(figsize=(32,8))
            avg_scores_pos = scores[:25000,0].reshape(-1, 100).mean(axis=1)
            avg_scores_neg = scores[:25000,1:4].mean(axis=1).reshape(-1, 100).mean(axis=1)
            plt.plot(avg_scores_pos, marker='o')
            plt.plot(avg_scores_neg, marker='*')
            plt.title("Averaged ITM Score (every 100 samples)")
            plt.xlabel("Batch (100 samples per batch)")
            plt.ylabel("Average Score")
            plt.legend(["Positive top1 Score", "Negative Score Top2-4 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_epochs_avg100_score_distribution.jpg"))


def plt_logging_list(logging_list, task="i2t", mode="tta"):
    # logging_list.keys() = dict_keys(['index', 'label', 'recall_type', 'sims', 'idxs_i2t', 'score', 'entropy', 'tta_coeffi', 'loss'])
    import numpy as np
    scores = np.array([item["score"] for item in logging_list])
    scores = scores.reshape(-1, scores.shape[-1]) # tta=(5000/25010, 4) eval=(5000/25010, 128)
    entropys = np.array([item["entropy"] for item in logging_list]).mean(axis=-1) #每个batch的entropy均值
    losses = np.array([item["loss"] for item in logging_list])
    
    if task == "i2t":
        if mode == "tta":
            plt.figure(figsize=(32,8))
            plt.plot(scores[:,0], alpha=0.7)
            plt.plot(scores[:,1:].mean(axis=1), alpha=0.7)
            plt.legend(["Positive top1 Score", "Negative Score top6-8 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_until_epochs_score_distribution.jpg"))

            # 计算每100个iter的score均值，输出5000/100维向量
            plt.figure(figsize=(32,8))
            avg_scores_pos = scores[:10*(scores.shape[0]//10),0].reshape(-1, 10).mean(axis=1)
            avg_scores_neg = scores[:10*(scores.shape[0]//10),1:].mean(axis=1).reshape(-1, 10).mean(axis=1)
            plt.plot(avg_scores_pos, marker='o')
            plt.plot(avg_scores_neg, marker='*')
            plt.title("Averaged ITM Score (every 10 samples)")
            plt.xlabel("X (10 samples per X)")
            plt.ylabel("Average Score")
            plt.legend(["Positive top1 Score", "Negative Score top6-8 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_until_epochs_avg10_score_distribution.jpg"))
        elif mode == "eval":
            assert False, "i2t eval mode not implemented yet, please use tta mode instead"
    elif task == "t2i":
        if mode == "tta":
            plt.figure(figsize=(32,8))
            plt.plot(scores[:,0], alpha=0.7)
            plt.plot(scores[:,1:].mean(axis=1), alpha=0.7)
            plt.legend(["Positive top1 Score", "Negative Score top2-4 Mean"])
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_until_epochs_score_distribution.jpg"))

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
            plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/{mode}_until_epochs_avg10_score_distribution.jpg"))
        elif mode == "eval":
            assert False, "t2i eval mode not implemented yet, please use tta mode instead"


def calculate_recall(topk_idx_sims, i2t_label):
    total_samples = len(i2t_label)
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    recall_types = []

    for i in range(total_samples):
        true_label = i2t_label[i]
        topk_indices = topk_idx_sims[i]
        r1_flag = False
        r5_flag = False
        r10_flag = False

        for label in true_label:
            if label in topk_indices[:1].tolist():
                r1_flag = True
            if label in topk_indices[:5].tolist():
                r5_flag = True
            if label in topk_indices[:10].tolist():
                r10_flag = True

        if r1_flag:
            recall_at_1 += 1
        if r5_flag:
            recall_at_5 += 1
        if r10_flag:
            recall_at_10 += 1

        if r1_flag:
            recall_types.append("recall@1")
        elif r5_flag:
            recall_types.append("recall@5")
        elif r10_flag:
            recall_types.append("recall@10")
        elif r1_flag==False and r5_flag==False and r10_flag==False:
            recall_types.append("negative sample")

    recall_at_1 /= total_samples
    recall_at_5 /= total_samples
    recall_at_10 /= total_samples

    return (recall_at_1, recall_at_5, recall_at_10), recall_types


## use DistributedSampler; shuffle=True
## top1 sample selection + 负样本采样计算softmax_entropy
def adapt_i2t_itm_score_v3(cfg, model, dataloader, task_cfg, optimizer, lr_scheduler, tta_cfg, sims_matrix_i2t, vit_feats, text_ids, text_atts, epoch=-1):
    logging.info("adapt_i2t_itm_score_v3: start")
    ## return outputs
    score_matrix_i2t = torch.full(
        (len(dataloader.dataset.image), len(dataloader.dataset.text)), -100.0
    ).to(model.device)
    scores_mat = torch.full(
        (len(dataloader.dataset.image), tta_cfg.k_tta), -100.0
    ).to(model.device)
    # labels_mat = torch.full(
    #     (len(dataloader.dataset.image), 5), -100.0
    # ).to(model.device)
    ## sampling stretegy
    k_test = task_cfg.k_test
    labels = dataloader.dataset.img2txt
    ## 获取metric标签用于可视化
    recall_types = find_recall_types(labels, sims_matrix_i2t, k_test=10)
    top128_sims, top128_idxs = sims_matrix_i2t.topk(k=cfg.run_cfg.k_test, dim=1)
    (recall1, recall5, recall10), recall_types_2 = calculate_recall(top128_idxs, labels)
    ## 采样正样本
    # pos_sample_range =  tta_cfg.pos_sample_range if hasattr(tta_cfg, "pos_sample_range") else [0, 1] # 正样本采样范围
    top1_sims, top1_idxs = sims_matrix_i2t.topk(k=1, dim=1) #正样本直接使用top1 TODO 使用top5采样一个正样本,但是这里相似度top5不一定就是5个label，所以还需要确定？
    top1_sims, top1_idxs =top1_sims[:,0], top1_idxs[:,0]
    ## 采样困难负样本
    neg_sample_range = tta_cfg.neg_sample_range if hasattr(tta_cfg, "neg_sample_range") else [32, 128] # 负样本采样范围
    neg_sims, neg_idxs = sample_neg_idxs(sims_matrix_i2t, tta_cfg.k_tta, k_test, neg_sample_range) # 负样本采样k_tta-1个, 根据score数值可视化差异确定采样范围
    ## 拼接正负样本
    sampled_sims_matrix_i2t = []
    sampled_sims_idx_i2t = []
    for i in range(sims_matrix_i2t.size(0)):
        sampled_sim = torch.concatenate((sims_matrix_i2t[i,top1_idxs[i]].reshape(-1), sims_matrix_i2t[i,neg_idxs[i]]))
        sampled_idx = torch.concatenate((top1_idxs[i].reshape(-1), neg_idxs[i]))
        sampled_sims_matrix_i2t.append(sampled_sim)
        sampled_sims_idx_i2t.append(sampled_idx)
    sampled_sims_matrix_i2t = torch.stack(sampled_sims_matrix_i2t) # shape=(5000, k_tta)
    sampled_sims_idx_i2t = torch.stack(sampled_sims_idx_i2t) # shape=(5000, k_tta)
    ## entropy coeffis
    if tta_cfg.top1_match_coeffi == True:
        tta_coeffis = compute_tta_coeffis(sims_matrix_i2t, sims_matrix_i2t.t(), k_test, tta_cfg.coeffi_i2t_temper, tta_cfg.coeffi_t2i_temper)
    else:
        tta_coeffis = [ torch.ones(1) for _ in range(sims_matrix_i2t.size(0)) ]
    ## score temperature
    score_temper_ = tta_cfg.score_temper if hasattr(tta_cfg, "score_temper") else 1.0
    logging.info("    number of batch sampling: {}".format(sampled_sims_matrix_i2t.size()))

    ## sample selection stretegy
    if tta_cfg.sample_selection == "top1":
        ss_idxs = find_inter_top1_sample_selection(sims_matrix_i2t, sims_matrix_i2t.t()) # 找到i2t和t2i互为top1的样本索引
        logging.info("    ss_idxs")
        # 只保留top1 sample selection的样本
        sims_matrix_i2t = sims_matrix_i2t[ss_idxs]
        # vit_feats = vit_feats.to(model.device)
        # vit_feats = vit_feats[ss_idxs]
        # logging.info("    vit_feats")
        # text_ids = text_ids[ss_idxs]
        # text_atts = text_atts[ss_idxs]
        labels = [labels[i] for i in ss_idxs]
        recall_types = [recall_types[i] for i in ss_idxs]
        logging.info("    labels recall_types")
        # top1_sims, top1_idxs = top1_sims[ss_idxs], top1_idxs[ss_idxs]
        # neg_sims, neg_idxs = neg_sims[ss_idxs], neg_idxs[ss_idxs]
        sampled_sims_matrix_i2t = sampled_sims_matrix_i2t[ss_idxs]
        sampled_sims_idx_i2t = sampled_sims_idx_i2t[ss_idxs]
        logging.info("    sampled_sims_matrix_i2t")
        tta_coeffis = [tta_coeffis[i] for i in ss_idxs]
        logging.info("    tta_coeffis")
        score_matrix_i2t_ = score_matrix_i2t[ss_idxs].cpu()
        logging.info("    score_matrix_i2t_")
        scores_mat_ = scores_mat[ss_idxs].cpu()
        logging.info("    scores_mat")
        # labels_mat_2_ = labels_mat_[ss_idxs]
    else:
        ss_idxs = torch.arange(0, sims_matrix_i2t.size(0)) # 全部样本
        score_matrix_i2t_ = score_matrix_i2t[ss_idxs].cpu()
        scores_mat_ = scores_mat[ss_idxs].cpu()
        # labels_mat_2_ = labels_mat_[ss_idxs]
    logging.info("    number of sample_selection: {}".format(len(ss_idxs)))

    ## tta_dataset & tta_dataloader
    tta_dataset = TTA_I2T_Dataset(
        tta_cfg,
        sims_matrix_i2t, sampled_sims_matrix_i2t, sampled_sims_idx_i2t,
        labels, recall_types,
        vit_feats[ss_idxs], text_ids, text_atts,
        tta_coeffis
    )
    logging.info("    number of tta_dataset: {}".format(len(tta_dataset)))
    ### DistributedSampler
    if cfg.run_cfg.distributed:
        sampler = DistributedSampler(
            tta_dataset,
            shuffle=True,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
    else:
        sampler = None
    ### Dataloader
    tta_dataloader = DataLoader(
        tta_dataset,
        batch_size=tta_cfg.tta_bs,
        num_workers=cfg.run_cfg.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=getattr(tta_dataset, "collater", None),
        drop_last=True
    )
    ### 预加载batch数据，加速数据通信
    tta_dataloader = PrefetchLoader(tta_dataloader)
    ### DDP设置sample.set_epoch(epoch)用于分布式sampler打乱数据
    ### 转化为无限iter(dataloader)
    if cfg.run_cfg.distributed:
        sampler.set_epoch(epoch)
    # tta_dataloader = IterLoader(tta_dataloader, use_distributed=cfg.run_cfg.distributed)
    logging.info("    number of tta_dataloader: {}".format(len(tta_dataloader)))

    ## training loop
    # score_matrix_list_ = []
    # scores_mat_list_ = []
    # labels_mat_list_ = []
    epoch_entropy_list = []
    epoch_loss_list = []
    logging_list = []
    grad_accum_num = 0

    logging.info("    start i2t itm adapt... ")
    start_time = time.time()
    model.eval()
    with torch.enable_grad():
        for iter, batch in enumerate(tta_dataloader):
            index = batch["index"]
            sims_all = batch["sims_all"]
            sims = batch["sims"]
            idxs_i2t = batch["idxs"]
            image_inputs = batch["image_inputs"].reshape(-1, batch["image_inputs"].size(-2), batch["image_inputs"].size(-1))
            text_ids_inputs = batch["text_ids"].reshape(-1, batch["text_ids"].size(-1))
            text_atts_inputs = batch["text_atts"].reshape(-1, batch["text_atts"].size(-1))
            tta_coeffi = batch["tta_coeffi"] # shape = bs,
            label = batch["label"] # shape = bs,
            recall_type = batch["recall_type"] # list of str  # shape = bs,

            logits = model.compute_itm_logits(
                image_inputs=image_inputs.to(model.device), #bs*k_tta,677,1408
                text_ids=text_ids_inputs.to(model.device), #bs*k_tta,35
                text_atts=text_atts_inputs.to(model.device), #bs*k_tta,35
            ).float() # logits.shape=bs*k_tta, 2
            logits = logits.reshape(-1, tta_cfg.k_tta, 2) # logits.shape=bs, k_tta, 2
            score = logits[..., 1] # score.shape=bs, k_tta

            ## score = itm_score + cos_sim
            for i, idx in enumerate(index):
                score_matrix_i2t_[idx.int(), idxs_i2t[i].int()] = score[i].detach().cpu() + sims.reshape(-1, tta_cfg.k_tta)[i].detach().cpu()
                scores_mat_[idx.int(), ...] = score[i].detach().cpu()
            # score_matrix_list_.append(score.detach().cpu() + sims.reshape(-1, tta_cfg.k_tta).detach().cpu())
            # scores_mat_list_.append(score.detach().cpu())
            # labels_mat_list_.append(label.detach().cpu())

            ## uncertainty
            uncertainty = torch.Tensor([1.]*score.size(0)).to(model.device) # shape = bs,
            if getattr(tta_cfg, "is_uncertainty", False) == True:
                # uncertainty = torch.Tensor([1.]*score.size(0)).to(model.device)
                # uncertainty = nn.functional.kl_div(sims.reshape(-1, tta_cfg.k_tta).to(model.device), score)
                uncertainty = nn.functional.kl_div(score, sims.reshape(-1, tta_cfg.k_tta).to(model.device)) # shape = bs,
                
            ## coeffi
            tta_coeffi = tta_coeffi.to(model.device)
            ## temperature
            score_temper = torch.Tensor([score_temper_]).to(model.device)
            ## adapt
            ### pos-neg softmax entropy
            entropy = -(F.softmax(score * score_temper, dim=-1) * F.log_softmax(score * score_temper, dim=-1)).sum(-1) # score * temper
            ### itm proba sigmoid entropy
            # entropy = (-(F.sigmoid(score * score_temper) * F.log(F.sigmoid(score * score_temper)))).sum(-1)
            loss = uncertainty * entropy / tta_coeffi
            loss = loss.mean()
            loss = loss / tta_cfg.grad_accum_bs
            loss.backward()
            grad_accum_num += 1
            if grad_accum_num >= tta_cfg.grad_accum_bs or iter+1 >= len(tta_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                grad_accum_num = 0
                if lr_scheduler is not None:
                    lr_scheduler.step()

            ## log_iters logging
            epoch_entropy_list.append(entropy.mean().detach().cpu().numpy())
            epoch_loss_list.append(loss.detach().cpu().numpy() * tta_cfg.grad_accum_bs)
            if 1:# if (iter+1) % tta_cfg.log_iters == 0 or iter+1 >= len(tta_dataloader):
                if lr_scheduler is not None:
                    lr = lr_scheduler.get_last_lr()[0]
                else:
                    lr = optimizer.param_groups[0]['lr']
                logging.info(f"[ITM ADAPT rank{get_rank()}] Iteration: {iter}, Iter Entropy Mean: {entropy.mean().detach().cpu().numpy()}, Iter Loss: {loss.detach().cpu().numpy()*tta_cfg.grad_accum_bs}, Learning Rate: {lr}")
            ## logging json
            logging_list.append({
                "index" : index.detach().cpu().numpy().tolist(), # index=ss_idxs[index]
                "label" : label,
                "recall_type" : recall_type,
                "sims" : sims.reshape(-1, tta_cfg.k_tta).detach().cpu().numpy().tolist(),
                "idxs_i2t" : idxs_i2t.reshape(-1, tta_cfg.k_tta).detach().cpu().numpy().tolist(),
                "score" : score.detach().cpu().numpy().tolist(),
                "entropy" : entropy.detach().cpu().numpy().tolist(),
                "tta_coeffi": tta_coeffi.detach().cpu().numpy().tolist(),
                "uncertainty": uncertainty.detach().cpu().numpy().tolist(),
                "loss" : loss.detach().cpu().numpy() * tta_cfg.grad_accum_bs,
                "lr": lr,
            })

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("    i2t itm adapt time: {}".format(total_time_str))

    # ## torch.distributed
    score_matrix_i2t[ss_idxs] = score_matrix_i2t_.to(model.device)
    scores_mat[ss_idxs] = scores_mat_.to(model.device)
    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            scores_mat, op=torch.distributed.ReduceOp.SUM
        )
    # score_matrix_list_ = torch.cat(score_matrix_list_, dim=0).to(model.device)
    # scores_mat_list_ = torch.cat(scores_mat_list_, dim=0).to(model.device)
    # labels_mat_list_ = torch.cat(labels_mat_list_, dim=0).to(model.device)
    # dist.all_gather(score_matrix_i2t, score_matrix_list_)
    # dist.all_gather(scores_mat, scores_mat_list_)
    # dist.all_gather(labels_mat, labels_mat_list_)
    # score_matrix_i2t = torch.cat(score_matrix_i2t, dim=0).numpy()
    # scores_mat = torch.cat(scores_mat, dim=0).numpy()
    # labels_mat = torch.cat(labels_mat, dim=0).numpy()

    # ## save epoch logging
    # logging_list_json = json.dumps(logging_list)
    # json_path = os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/tta_epoch{epoch}_logging_list.json")
    # with open(json_path, "w") as f:
    #     json.dump(logging_list_json, f)
    # ## plt entropy
    # plt.figure()
    # plt.plot(epoch_entropy_list)
    # plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/tta_epoch{epoch}_entropy.jpg"))
    # ## plt loss
    # plt.figure()
    # plt.plot(epoch_loss_list)
    # plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/tta_epoch{epoch}_loss.jpg"))
    # ## plt score
    # if is_main_process():
    #     plt.figure(figsize=(32,8))
    #     plt.plot(scores_mat[:,0].cpu().detach().numpy(), alpha=0.7)
    #     plt.plot(scores_mat[:,1:].cpu().detach().numpy().mean(axis=1), alpha=0.7)
    #     plt.legend(["Positive top1 Score", "Negative top5-8 Score Mean"])
    #     plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/tta_epoch{epoch}_score_distribution.jpg"))

    logging.info("adapt_i2t_itm_score_v3: end")
    return score_matrix_i2t.cpu().detach().numpy(), scores_mat.cpu().detach().numpy(), logging_list#, labels_mat.cpu().detach().numpy()

## use DistributedSampler; shuffle=True
## top1 sample selection + 负样本采样计算softmax_entropy
def adapt_t2i_itm_score_v3(cfg, model, dataloader, task_cfg, optimizer, tta_cfg, sims_matrix_t2i, vit_feats, text_ids, text_atts, epoch=-1):
    logging.info("adapt_t2i_itm_score_v3: start")
    ## return outputs
    score_matrix_t2i = torch.full(
        ( len(dataloader.dataset.text), len(dataloader.dataset.image)), -100.0
    ).to(model.device)
    scores_mat = torch.full(
        (len(dataloader.dataset.text), tta_cfg.k_tta), -100.0
    ).to(model.device)
    # labels_mat = torch.full(
    #     (len(dataloader.dataset.image), 5), -100.0
    # ).to(model.device)
    ## sampling stretegy
    k_test = task_cfg.k_test
    labels = dataloader.dataset.txt2img
    ## 获取metric标签用于可视化
    recall_types = find_recall_types(labels, sims_matrix_t2i, k_test=10)
    ## 采样正样本
    # pos_sample_range =  tta_cfg.pos_sample_range if hasattr(tta_cfg, "pos_sample_range") else [0, 1] # 正样本采样范围
    top1_sims, top1_idxs = sims_matrix_t2i.topk(k=1, dim=1) #正样本直接使用top1 TODO 使用top5采样一个正样本,但是这里相似度top5不一定就是5个label，所以还需要确定？
    top1_sims, top1_idxs =top1_sims[:,0], top1_idxs[:,0]
    ## 采样困难负样本
    neg_sample_range = tta_cfg.neg_sample_range if hasattr(tta_cfg, "neg_sample_range") else [32, 128] # 负样本采样范围
    neg_sims, neg_idxs = sample_neg_idxs(sims_matrix_t2i, tta_cfg.k_tta, k_test, neg_sample_range) # 负样本采样k_tta-1个, 根据score数值可视化差异确定采样范围
    ## 拼接正负样本
    sampled_sims_matrix_t2i = []
    sampled_sims_idx_t2i = []
    for i in range(sims_matrix_t2i.size(0)):
        sampled_sim = torch.concatenate((sims_matrix_t2i[i,top1_idxs[i]].reshape(-1), sims_matrix_t2i[i,neg_idxs[i]]))
        sampled_idx = torch.concatenate((top1_idxs[i].reshape(-1), neg_idxs[i]))
        sampled_sims_matrix_t2i.append(sampled_sim)
        sampled_sims_idx_t2i.append(sampled_idx)
    sampled_sims_matrix_t2i = torch.stack(sampled_sims_matrix_t2i) # shape=(5000, k_tta)
    sampled_sims_idx_t2i = torch.stack(sampled_sims_idx_t2i) # shape=(5000, k_tta)
    ## entropy coeffis
    if tta_cfg.top1_match_coeffi == True:
        tta_coeffis = compute_tta_coeffis(sims_matrix_t2i, sims_matrix_t2i.t(), k_test, tta_cfg.coeffi_i2t_temper, tta_cfg.coeffi_t2i_temper)
    else:
        tta_coeffis = [ torch.ones(1) for _ in range(sims_matrix_t2i.size(0)) ]
    ## score temperature
    score_temper_ = tta_cfg.score_temper if hasattr(tta_cfg, "score_temper") else 1.0

    ## sample selection stretegy
    if tta_cfg.sample_selection == "top1":
        ss_idxs = find_inter_top1_sample_selection(sims_matrix_t2i, sims_matrix_t2i.t()) # 找到t2i和i2t互为top1的样本索引
        # 只保留top1 sample selection的样本
        # sims_matrix_t2i = sims_matrix_t2i[ss_idxs]
        # vit_feats = vit_feats[ss_idxs]
        text_ids = text_ids[ss_idxs]
        text_atts = text_atts[ss_idxs]
        labels = [labels[i] for i in ss_idxs]
        recall_types = [recall_types[i] for i in ss_idxs]
        # top1_sims, top1_idxs = top1_sims[ss_idxs], top1_idxs[ss_idxs]
        # neg_sims, neg_idxs = neg_sims[ss_idxs], neg_idxs[ss_idxs]
        sampled_sims_matrix_t2i = sampled_sims_matrix_t2i[ss_idxs]
        tta_coeffis = [tta_coeffis[i] for i in ss_idxs]
        score_matrix_t2i_ = score_matrix_t2i[ss_idxs].cpu()
        scores_mat_ = scores_mat[ss_idxs].cpu()
        # labels_mat_2_ = labels_mat_[ss_idxs]
    else:
        ss_idxs = torch.arange(0, sims_matrix_t2i.size(0)) # 全部样本
        score_matrix_t2i_ = score_matrix_t2i[ss_idxs].cpu()
        scores_mat_ = scores_mat[ss_idxs].cpu()
        # labels_mat_2_ = labels_mat_[ss_idxs]
    logging.info("    number of sample_selection: {}".format(len(ss_idxs)))

    ## tta_dataset & tta_dataloader
    tta_dataset = TTA_T2I_Dataset(
        tta_cfg,
        sampled_sims_matrix_t2i, sampled_sims_idx_t2i,
        labels, recall_types,
        vit_feats, text_ids, text_atts,
        tta_coeffis
    )
    logging.info("    number of tta_dataset: {}".format(len(tta_dataset)))
    ### DistributedSampler
    if cfg.run_cfg.distributed:
        sampler = DistributedSampler(
            tta_dataset,
            shuffle=True,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
    else:
        sampler = None
    ### Dataloader
    tta_dataloader = DataLoader(
        tta_dataset,
        batch_size=tta_cfg.tta_bs,
        num_workers=cfg.run_cfg.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=getattr(tta_dataset, "collater", None),
        drop_last=True
    )
    ### 预加载batch数据，加速数据通信
    tta_dataloader = PrefetchLoader(tta_dataloader)
    ### DDP设置sample.set_epoch(epoch)用于分布式sampler打乱数据
    ### 转化为无限iter(dataloader)
    if cfg.run_cfg.distributed:
        sampler.set_epoch(epoch)
    # tta_dataloader = IterLoader(tta_dataloader, use_distributed=cfg.run_cfg.distributed)
    logging.info("    number of tta_dataloader: {}".format(len(tta_dataloader)))

    ## training loop
    # score_matrix_list_ = []
    # scores_mat_list_ = []
    # labels_mat_list_ = []
    epoch_entropy_list = []
    epoch_loss_list = []
    logging_list = []
    grad_accum_num = 0

    logging.info("    start t2i itm adapt... ")
    start_time = time.time()
    model.eval()
    with torch.enable_grad():
        for iter, batch in enumerate(tta_dataloader):
            index = batch["index"]
            sims = batch["sims"]
            idxs_t2i = batch["idxs"]
            image_inputs = batch["image_inputs"].reshape(-1, batch["image_inputs"].size(-2), batch["image_inputs"].size(-1))
            text_ids_inputs = batch["text_ids"].reshape(-1, batch["text_ids"].size(-1))
            text_atts_inputs = batch["text_atts"].reshape(-1, batch["text_atts"].size(-1))
            tta_coeffi = batch["tta_coeffi"]
            label = batch["label"]
            recall_type = batch["recall_type"] #list of str

            logits = model.compute_itm_logits(
                image_inputs=image_inputs.to(model.device), #bs*k_tta,677,1408
                text_ids=text_ids_inputs.to(model.device), #bs*k_tta,35
                text_atts=text_atts_inputs.to(model.device), #bs*k_tta,35
            ).float() # logits.shape=bs*k_tta, 2
            logits = logits.reshape(-1, tta_cfg.k_tta, 2) # logits.shape=bs, k_tta, 2
            score = logits[..., 1] # score.shape=bs, k_tta

            ## score = itm_score + cos_sim
            for i, idx in enumerate(index):
                score_matrix_t2i_[idx.int(), idxs_t2i[i].int()] = score[i].detach().cpu() + sims.reshape(-1, tta_cfg.k_tta)[i].detach().cpu()
                scores_mat_[idx.int(), ...] = score[i].detach().cpu()
            # score_matrix_list_.append(score.detach().cpu() + sims.reshape(-1, tta_cfg.k_tta).detach().cpu())
            # scores_mat_list_.append(score.detach().cpu())
            # labels_mat_list_.append(label.detach().cpu())

            ## coeffi
            tta_coeffi = tta_coeffi.to(model.device)
            ## temperature
            score_temper = torch.Tensor([score_temper_]).to(model.device)
            ## adapt
            ### pos-neg softmax entropy
            entropy = -(F.softmax(score * score_temper, dim=-1) * F.log_softmax(score * score_temper, dim=-1)).sum(-1) # score * temper
            ### itm proba sigmoid entropy
            # entropy = (-(F.sigmoid(score * score_temper) * F.log(F.sigmoid(score * score_temper)))).sum(-1)
            loss = entropy / tta_coeffi
            loss = loss.mean()
            loss = loss / tta_cfg.grad_accum_bs
            loss.backward()
            grad_accum_num += 1
            if grad_accum_num >= tta_cfg.grad_accum_bs or iter+1 >= len(tta_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                grad_accum_num = 0

            ## logging json
            logging_list.append({
                "index" : index.detach().cpu().numpy().tolist(),
                "label" : label,
                "recall_type" : recall_type,
                "sims" : sims.reshape(-1, tta_cfg.k_tta).detach().cpu().numpy().tolist(),
                "idxs_t2i" : idxs_t2i.reshape(-1, tta_cfg.k_tta).detach().cpu().numpy().tolist(),
                "score" : score.detach().cpu().numpy().tolist(),
                "entropy" : entropy.detach().cpu().numpy().tolist(),
                "tta_coeffi": tta_coeffi.detach().cpu().numpy().tolist(),
                "loss" : loss.detach().cpu().numpy() * tta_cfg.grad_accum_bs,
            })
            ## log_iters logging
            epoch_entropy_list.append(entropy.mean().detach().cpu().numpy())
            epoch_loss_list.append(loss.detach().cpu().numpy() * tta_cfg.grad_accum_bs)
            if 1:# if (iter+1) % tta_cfg.log_iters == 0 or iter+1 >= len(tta_dataloader):
                logging.info(f"[ITM ADAPT rank{get_rank()}] Iteration: {iter}, Iter Entropy Mean: {entropy.mean().detach().cpu().numpy()}, Iter Loss: {loss.detach().cpu().numpy()*tta_cfg.grad_accum_bs}, Learning Rate: {optimizer.param_groups[0]['lr']}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("    t2i itm adapt time: {}".format(total_time_str))

    # ## torch.distributed
    score_matrix_t2i[ss_idxs] = score_matrix_t2i_.to(model.device)
    scores_mat[ss_idxs] = scores_mat_.to(model.device)
    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            scores_mat, op=torch.distributed.ReduceOp.SUM
        )
    # score_matrix_list_ = torch.cat(score_matrix_list_, dim=0).to(model.device)
    # scores_mat_list_ = torch.cat(scores_mat_list_, dim=0).to(model.device)
    # labels_mat_list_ = torch.cat(labels_mat_list_, dim=0).to(model.device)
    # dist.all_gather(score_matrix_t2i, score_matrix_list_)
    # dist.all_gather(scores_mat, scores_mat_list_)
    # dist.all_gather(labels_mat, labels_mat_list_)
    # score_matrix_t2i = torch.cat(score_matrix_t2i, dim=0).numpy()
    # scores_mat = torch.cat(scores_mat, dim=0).numpy()
    # labels_mat = torch.cat(labels_mat, dim=0).numpy()

    # ## save epoch logging
    # logging_list_json = json.dumps(logging_list)
    # json_path = os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/tta_epoch{epoch}_logging_list.json")
    # with open(json_path, "w") as f:
    #     json.dump(logging_list_json, f)
    # ## plt entropy
    # plt.figure()
    # plt.plot(epoch_entropy_list)
    # plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/tta_epoch{epoch}_entropy.jpg"))
    # ## plt loss
    # plt.figure()
    # plt.plot(epoch_loss_list)
    # plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/rank{get_rank()}/tta_epoch{epoch}_loss.jpg"))
    # ## plt score
    # if is_main_process():
    #     plt.figure(figsize=(32,8))
    #     plt.plot(scores_mat[:,0].cpu().detach().numpy(), alpha=0.7)
    #     plt.plot(scores_mat[:,1:].cpu().detach().numpy().mean(axis=1), alpha=0.7)
    #     plt.legend(["Positive top1 Score", "Negative top2-4 Score Mean"])
    #     plt.savefig(os.path.join(registry.get_path("output_dir"), f"result/tta_epoch{epoch}_score_distribution.jpg"))

    logging.info("adapt_t2i_itm_score_v3: end")
    return score_matrix_t2i.cpu().detach().numpy(), scores_mat.cpu().detach().numpy(), logging_list#, labels_mat.cpu().detach().numpy()


