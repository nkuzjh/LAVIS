"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
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

class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        except:
            tokenizer = BertTokenizer.from_pretrained("/home/zhh/ssd/excute/deeplearning/projects/checkpoints/bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        try:
            encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        except:
            encoder_config = BertConfig.from_pretrained("/home/zhh/ssd/excute/deeplearning/projects/checkpoints/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        try:
            Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        except:
            Qformer = BertLMHeadModel.from_pretrained("/home/zhh/ssd/excute/deeplearning/projects/checkpoints/bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
#         elif model_name == "eva2_clip_L":
#             visual_encoder = create_eva2_vit_L(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def get_optimizer_params(self, weight_decay, lr_scale=1):

        vit_num_layers = self.visual_encoder.get_num_layer()
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                group_name = "vit_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        # import json
        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


@torch.no_grad()
def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

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

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
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

def compute_sim_matrix_worerank(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

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

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        score_matrix_i2t[start + i, topk_idx] = topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        score_matrix_t2i[start + i, topk_idx] = topk_sim

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

def compute_i2t_sim_matrix_adapt(model, data_loader, optimizer, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for i2t tta evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    logging.info("    text features...")
    with torch.no_grad():
        for i in range(0, num_text, text_bs):#bs=256 all=25010
            print("\r text_batch_i: ", i, end="")
            text = texts[i : min(num_text, i + text_bs)]
            text_input = model.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(model.device)
            text_feat = model.forward_text(text_input)#text_input.input_ids.shape=bs,35
            text_embed = F.normalize(model.text_proj(text_feat))#text_feat.shape=bs,768 text_embed.shape=bs,256
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)#text_input.input_ids.shape=bs,35
            text_atts.append(text_input.attention_mask)#text_input.attention_mask.shape=bs,35

        text_embeds = torch.cat(text_embeds, dim=0) #25010,256
        text_ids = torch.cat(text_ids, dim=0)#25010,35
        text_atts = torch.cat(text_atts, dim=0)#25010,35
    print()

    vit_feats = []
    image_embeds = []
    sims_matrix = []
    logging.info("    image features...")
    for i, samples in enumerate(data_loader):#bs=16 #len(data_loader)=313 all==
        print("\r image_batch_i: ", i*len(samples["image"]), end="")
        image = samples["image"]

        image = image.to(model.device) #bs,3,364,364
        image_feat, vit_feat = model.forward_image(image) #image_feat.shape=bs,32,768 vit_feat.shape=bs,677,1408
        image_embed = model.vision_proj(image_feat)#bs,32,768->bs,32,256
        image_embed = F.normalize(image_embed, dim=-1)

        # compute sim_matirx per-batch
        sim_q2t = image_embed @ text_embeds.t() #text_embeds.shape=25010,256 #image_embed.shape=bs,32,256 sim_q2t.shape=bs,32,25010
        # sim_i2t, _ = sim_q2t.max(0)#sim_i2t.shape=32,25010
        sim_i2t, _ = sim_q2t.max(1)#sim_i2t.shape=bs,25010
        _sim_i2t = sim_i2t / model.temp
        # test time adapt
        loss = softmax_entropy(_sim_i2t).mean(0) #softmax_entropy(sim_i2t).shape=bs
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)
        sims_matrix.append(sim_i2t)
    print()

    vit_feats = torch.cat(vit_feats, dim=0)#vit_feats.shape=5000,677,1408
    image_embeds = torch.cat(image_embeds, dim=0)#image_embeds.shape=5000,32,256
    sims_matrix = torch.cat(sims_matrix, dim=0)#sims_matrix.shaope=5000,25010

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    model.eval()
    with torch.no_grad():
        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 100, header) # sims_matrix.shape=5000,25010
        ): # 遍历每个image与25010个text的sim_matrix
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0) #sims.shape=1,25010
            image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device) # vit_feats[start + i].shape=677,1408 image_inputs.shape=128,677,1408
            score = model.compute_itm(
                image_inputs=image_inputs, # 将一张图片的vit_feat repeat 128次，对齐topk(k=k_test)个候选text
                text_ids=text_ids[topk_idx], #topk(k=k_test)个候选text
                text_atts=text_atts[topk_idx], #topk(k=k_test)个候选text的attention_mask
            ).float()
            score_matrix_i2t[start + i, topk_idx] = score + topk_sim # similarity + itm_score最终预测得分

        # sims_matrix = sims_matrix.t()
        # score_matrix_t2i = torch.full(
        #     (len(texts), len(data_loader.dataset.image)), -100.0
        # ).to(model.device)

        # step = sims_matrix.size(0) // num_tasks + 1
        # start = rank * step
        # end = min(sims_matrix.size(0), start + step)

        # for i, sims in enumerate(
        #     metric_logger.log_every(sims_matrix[start:end], 50, header)
        # ):
        #     topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        #     image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        #     score = model.compute_itm(
        #         image_inputs=image_inputs,
        #         text_ids=text_ids[start + i].repeat(k_test, 1),
        #         text_atts=text_atts[start + i].repeat(k_test, 1),
        #     ).float()
        #     score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        # torch.distributed.all_reduce(
        #     score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        # )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("i2t tta Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy()#, score_matrix_t2i.cpu().numpy()

def compute_t2i_sim_matrix_adapt(model, data_loader, optimizer, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for t2i tta evaluation...")
    start_time = time.time()

    vit_feats = []
    image_embeds = []
    logging.info("    image features...")
    with torch.no_grad():
        for i, samples in enumerate(data_loader):
            print("\r image_batch_i: ", i*len(samples["image"]), end="")
            image = samples["image"]

            image = image.to(model.device)
            image_feat, vit_feat = model.forward_image(image)
            image_embed = model.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)

            vit_feats.append(vit_feat.cpu())
            image_embeds.append(image_embed)
        print()

        vit_feats = torch.cat(vit_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    sims_matrix = []
    logging.info("    text features...")
    for i in range(0, num_text, text_bs):
        print("\r text_batch_i: ", i, end="")
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

        # compute sim_matirx per-batch
        # print(text_embed.shape) # bs,256
        # print(image_embeds.shape) # 5000,32,256
        sim_q2i = (image_embeds @ text_embed.t()).permute(2,1,0) # 5000,32,256 @ 256,bs -> 5000,32,bs -> bs,32,5000
        # sim_t2i, _ = sim_q2i.max(0)
        sim_t2i, _ = sim_q2i.max(1)
        _sim_t2i = sim_t2i / model.temp
        # test time adapt
        loss = softmax_entropy(_sim_t2i).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
        sims_matrix.append(sim_t2i)
    print()

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    # sims_matrix = torch.stack(sims_matrix, dim=0)
    sims_matrix = torch.cat(sims_matrix, dim=0)

    # score_matrix_i2t = torch.full(
    #     (len(data_loader.dataset.image), len(texts)), -100.0
    # ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # step = sims_matrix.size(0) // num_tasks + 1
    # start = rank * step
    # end = min(sims_matrix.size(0), start + step)

    # for i, sims in enumerate(
    #     metric_logger.log_every(sims_matrix[start:end], 50, header)
    # ):
    #     topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
    #     image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
    #     score = model.compute_itm(
    #         image_inputs=image_inputs,
    #         text_ids=text_ids[topk_idx],
    #         text_atts=text_atts[topk_idx],
    #     ).float()
    #     score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    # sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    model.eval()
    with torch.no_grad():
        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 100, header)
        ):
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
        # torch.distributed.all_reduce(
        #     score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        # )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("t2i tta Evaluation time {}".format(total_time_str))

    return score_matrix_t2i.cpu().numpy() #score_matrix_i2t.cpu().numpy(),

def compute_i2t_sim_matrix_adapt_worerank(model, data_loader, optimizer, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for i2t tta evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    logging.info("    text features...")
    with torch.no_grad():
        for i in range(0, num_text, text_bs):#bs=256 all=25010
            print("\r text_batch_i: ", i, end="")
            text = texts[i : min(num_text, i + text_bs)]
            text_input = model.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(model.device)
            text_feat = model.forward_text(text_input)#text_input.input_ids.shape=bs,35
            text_embed = F.normalize(model.text_proj(text_feat))#text_feat.shape=bs,768 text_embed.shape=bs,256
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)#text_input.input_ids.shape=bs,35
            text_atts.append(text_input.attention_mask)#text_input.attention_mask.shape=bs,35

        text_embeds = torch.cat(text_embeds, dim=0) #25010,256
        text_ids = torch.cat(text_ids, dim=0)#25010,35
        text_atts = torch.cat(text_atts, dim=0)#25010,35
    print()

    vit_feats = []
    image_embeds = []
    sims_matrix = []
    logging.info("    image features...")
    for i, samples in enumerate(data_loader):#bs=16 #len(data_loader)=313 all==
        print("\r image_batch_i: ", i*len(samples["image"]), end="")
        image = samples["image"]

        image = image.to(model.device) #bs,3,364,364
        image_feat, vit_feat = model.forward_image(image) #image_feat.shape=bs,32,768 vit_feat.shape=bs,677,1408
        image_embed = model.vision_proj(image_feat)#bs,32,768->bs,32,256
        image_embed = F.normalize(image_embed, dim=-1)

        # compute sim_matirx per-batch
        sim_q2t = image_embed @ text_embeds.t() #text_embeds.shape=25010,256 #image_embed.shape=bs,32,256 sim_q2t.shape=bs,32,25010
        # sim_i2t, _ = sim_q2t.max(0)#sim_i2t.shape=32,25010
        sim_i2t, _ = sim_q2t.max(1)#sim_i2t.shape=bs,25010
        sim_i2t = sim_i2t / model.temp
        # test time adapt
        loss = softmax_entropy(sim_i2t).mean(0) #softmax_entropy(sim_i2t).shape=bs
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)
        sims_matrix.append(sim_i2t)
    print()

    vit_feats = torch.cat(vit_feats, dim=0)#vit_feats.shape=5000,677,1408
    image_embeds = torch.cat(image_embeds, dim=0)#image_embeds.shape=5000,32,256
    sims_matrix = torch.cat(sims_matrix, dim=0)#sims_matrix.shaope=5000,25010

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    model.eval()
    with torch.no_grad():
        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 100, header) # sims_matrix.shape=5000,25010
        ): # 遍历每个image与25010个text的sim_matrix
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0) #sims.shape=1,25010 #k_test=128
            score_matrix_i2t[start + i, topk_idx] = topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("i2t tta Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy()#, score_matrix_t2i.cpu().numpy()

def compute_t2i_sim_matrix_adapt_worerank(model, data_loader, optimizer, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for t2i tta evaluation...")
    start_time = time.time()

    vit_feats = []
    image_embeds = []
    logging.info("    image features...")
    with torch.no_grad():
        for i, samples in enumerate(data_loader):
            print("\r image_batch_i: ", i*len(samples["image"]), end="")
            image = samples["image"]

            image = image.to(model.device)
            image_feat, vit_feat = model.forward_image(image)
            image_embed = model.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)

            vit_feats.append(vit_feat.cpu())
            image_embeds.append(image_embed)
        print()

        vit_feats = torch.cat(vit_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    sims_matrix = []
    logging.info("    text features...")
    for i in range(0, num_text, text_bs):
        print("\r text_batch_i: ", i, end="")
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

        # compute sim_matirx per-batch
        # print(text_embed.shape) # bs,256
        # print(image_embeds.shape) # 5000,32,256
        sim_q2i = (image_embeds @ text_embed.t()).permute(2,1,0) # 5000,32,256 @ 256,bs -> 5000,32,bs -> bs,32,5000
        # sim_t2i, _ = sim_q2i.max(0)
        sim_t2i, _ = sim_q2i.max(1)
        sim_t2i = sim_t2i / model.temp
        # test time adapt
        loss = softmax_entropy(sim_t2i).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
        sims_matrix.append(sim_t2i)
    print()

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    # sims_matrix = torch.stack(sims_matrix, dim=0)
    sims_matrix = torch.cat(sims_matrix, dim=0)

    # score_matrix_i2t = torch.full(
    #     (len(data_loader.dataset.image), len(texts)), -100.0
    # ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()

    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    model.eval()
    with torch.no_grad():
        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 100, header)
        ):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
            score_matrix_t2i[start + i, topk_idx] = topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("t2i tta Evaluation time {}".format(total_time_str))

    return score_matrix_t2i.cpu().numpy() #score_matrix_i2t.cpu().numpy(),

def compute_i2t_sim_matrix_adapt_zhh(model, data_loader, optimizer, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation itm re-rank:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        logging.info("    text features...")
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
            print("\r text_batch_i: ", i, end="")
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

        logging.info("    image features...")
        vit_feats = []
        image_embeds = []
        for i, samples in enumerate(data_loader):
            print("\r image_batch_i: ", i*len(samples["image"]), end="")
            image = samples["image"]

            image = image.to(model.device)
            image_feat, vit_feat = model.forward_image(image)
            image_embed = model.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)

            vit_feats.append(vit_feat.cpu())
            image_embeds.append(image_embed)

        vit_feats = torch.cat(vit_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        logging.info("    similarity matrix...") # 这里F.normaliza()后的矩阵相乘@ 就是cosine_similarity
        sims_matrix = []
        for image_embed in image_embeds: # 5000,32,256
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010 -> 25010
            # sim_i2t = sim_i2t / model.temp
            sims_matrix.append(sim_i2t)
        sims_matrix = torch.stack(sims_matrix, dim=0) #5000,25010

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()

    model.train()

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 100, header)
    ): # 遍历每个image与25010个text的sim_matrix
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0) #sims.shape=1,25010 topk_sim.shape=128 topk_idx=top128_idx
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device) # vit_feats[start + i].shape=677,1408 image_inputs.shape=128,677,1408
        score = model.compute_itm(
            image_inputs=image_inputs, # 将一张图片的vit_feat repeat 128次，对齐topk(k=k_test)个候选text #128,677,1408
            text_ids=text_ids[topk_idx], #topk(k=k_test)个候选text #128,35
            text_atts=text_atts[topk_idx], #topk(k=k_test)个候选text的attention_mask 128,35
        ).float() # score.shape=128

        # itm tta with uncertainty
        # loss_ent = zhh_softmax_entropy(score)#.mean(0)
        loss_entropy_topk_gallery = -(F.softmax(score) * F.log_softmax(score))
        # loss_entropy_uncertainty = ( ( torch.exp((1 + sims[topk_idx])/2) - 1 )  * loss_entropy_topk_gallery ).mean()
        # loss_entropy_uncertainty = ( (1 + sims[topk_idx])/2  * loss_entropy_topk_gallery + (1 - (1 + sims[topk_idx])/2) ).mean()
        # loss_entropy_uncertainty = ( (1 + sims[topk_idx])/2  * loss_entropy_topk_gallery ).mean()
        loss_entropy_uncertainty = loss_entropy_topk_gallery.mean()
        loss_entropy_uncertainty.backward()
        optimizer.step()
        optimizer.zero_grad()

        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().detach().numpy()

def compute_t2i_sim_matrix_adapt_zhh(model, data_loader, optimizer, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation itm re-rank:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        logging.info("    text features...")
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
            print("\r text_batch_i: ", i, end="")
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

        logging.info("    image features...")
        vit_feats = []
        image_embeds = []
        for i, samples in enumerate(data_loader):
            print("\r image_batch_i: ", i*len(samples["image"]), end="")
            image = samples["image"]

            image = image.to(model.device)
            image_feat, vit_feat = model.forward_image(image)
            image_embed = model.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)

            vit_feats.append(vit_feat.cpu())
            image_embeds.append(image_embed)

        vit_feats = torch.cat(vit_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        logging.info("    similarity matrix...")
        sims_matrix = []
        for image_embed in image_embeds: # 5000,32,256
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010
            # sim_i2t = sim_i2t / model.temp
            sims_matrix.append(sim_i2t) # 25010
        sims_matrix = torch.stack(sims_matrix, dim=0) #5000,25010
        sims_matrix = sims_matrix.t()

    model.train()

    score_matrix_t2i = torch.full(
            (len(texts), len(data_loader.dataset.image)), -100.0
        ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 500, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        # itm tta with uncertainty
        loss_entropy_topk_gallery = -(F.softmax(score) * F.log_softmax(score))
        # loss_entropy_uncertainty = ( (1 + sims[topk_idx])/2  * loss_entropy_topk_gallery + (1 - (1 + sims[topk_idx])/2) ).mean()
        # loss_entropy_uncertainty = ( (1 + sims[topk_idx])/2  * loss_entropy_topk_gallery ).mean()
        loss_entropy_uncertainty = loss_entropy_topk_gallery.mean()
        loss_entropy_uncertainty.backward()
        optimizer.step()
        optimizer.zero_grad()

        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_t2i.cpu().detach().numpy()

def compute_i2t_sim_matrix_adapt_zhh_topk(model, data_loader, optimizer,  **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation itm re-rank:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    logging.info("    text features...")
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []

    model.eval()
    with torch.no_grad():
        for i in range(0, num_text, text_bs):
            print("\r text_batch_i: ", i, end="")
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
        print()

        vit_feats = []
        image_embeds = []
        logging.info("    image features...")
        for i, samples in enumerate(data_loader):
            print("\r image_batch_i: ", i*len(samples["image"]), end="")
            image = samples["image"]

            image = image.to(model.device)
            image_feat, vit_feat = model.forward_image(image)
            image_embed = model.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)

            vit_feats.append(vit_feat.cpu())
            image_embeds.append(image_embed)
        print()

        vit_feats = torch.cat(vit_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        logging.info("    similarity matrix...") # 这里F.normaliza()后的矩阵相乘@ 就是cosine_similarity
        sims_matrix = []
        for image_embed in image_embeds: # 5000,32,256
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010 -> 25010
            sim_i2t = sim_i2t / model.temp #model.temp=0.0295 #im_i2t.max(),sim_i2t.min(),sim_i2t.mean()=(tensor(0.5524, device='cuda:0'), tensor(0.1366, device='cuda:0'), tensor(0.2560, device='cuda:0')) #_sim_i2t.max(),_sim_i2t.min(),_sim_i2t.mean()=(tensor(18.7249, device='cuda:0'), tensor(4.6288, device='cuda:0'), tensor(8.6786, device='cuda:0'))
            sims_matrix.append(sim_i2t)
        sims_matrix_i2t = torch.stack(sims_matrix, dim=0) #5000,25010
        sims_matrix_t2i = sims_matrix_i2t.t()

        # import numpy as np
        # np.save("sim_matrix_i2t.npy", sims_matrix_i2t.detach().cpu().numpy())

        # num_tasks = dist_utils.get_world_size()
        # rank = dist_utils.get_rank()

        # logging.info("    compute topk index of similarity matrix...")
        # topk_idx_matrix_i2t = torch.full(
        #     (len(data_loader.dataset.image), k_test), fill_value=-100, dtype=torch.int64
        # ).to(model.device)
        # topk_idx_matrix_t2i = torch.full(
        #     (len(texts), k_test), fill_value=-100, dtype=torch.int64
        # ).to(model.device)

        # step = sims_matrix_i2t.size(0) // num_tasks + 1
        # start = rank * step
        # end = min(sims_matrix_i2t.size(0), start + step)
        # for i, sims in enumerate(sims_matrix_i2t[start:end]): # 遍历每个image与25010个text的sim_matrix
        #     topk_sim, topk_idx = sims.topk(k=k_test, dim=0) #sims.shape=1,25010 topk_sim.shape=128 topk_idx=top128_idx
        #     topk_idx_matrix_i2t[start + i, ...] = topk_idx

        # step = sims_matrix_t2i.size(0) // num_tasks + 1
        # start = rank * step
        # end = min(sims_matrix_t2i.size(0), start + step)
        # for i, sims in enumerate(sims_matrix_t2i[start:end]): # 遍历每个image与25010个text的sim_matrix
        #     topk_sim, topk_idx = sims.topk(k=k_test, dim=0) #sims.shape=1,25010 topk_sim.shape=128 topk_idx=top128_idx
        #     topk_idx_matrix_t2i[start + i, ...] = topk_idx

    # sumi=0
    # sumi2=0
    # for i in range(0,5000):
    #     if topk_idx_matrix_i2t[topk_idx_matrix_t2i[i,0]][0] != i:
    #         sumi+=1
    #     if topk_idx_matrix_t2i[topk_idx_matrix_i2t[i,0]][0] != i:
    #         sumi2+=1
    # print(sumi, sumi2)
    # # 714个t2i的top1与i2t的top1不一致；4286个t2i的top1与i2t的top1一致
    # # 4134个i2t的top1与t2i的top1不一致；866个i2t的top1与t2i的top1一致

    torch.cuda.empty_cache()

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_i2t.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_i2t.size(0), start + step)
    itm_loss_backward_accum_bs = 64
    grad_accum_num = 0

    model.train()
    for i, sims_i2t in enumerate(
        metric_logger.log_every(sims_matrix_i2t[start:end], 50, header)
    ): # 遍历每个image与25010个text的sim_matrix
        topk_sim_i2t, topk_idx_i2t = sims_i2t.topk(k=k_test, dim=0) #sims.shape=25010 topk_sim.shape=128 topk_idx=top128_idx
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device) # vit_feats[i].shape=1,677,1408 image_inputs.shape=128,677,1408
        score = model.compute_itm(
            image_inputs=image_inputs.to(model.device), # 将一张图片的vit_feat repeat 128次，对齐topk(k=k_test)个候选text #128,677,1408
            text_ids=text_ids[topk_idx_i2t].to(model.device), #topk(k=k_test)个候选text #128,35
            text_atts=text_atts[topk_idx_i2t].to(model.device), #topk(k=k_test)个候选text的attention_mask 128,35
        ).float() # score.shape=128

        proba_top1_sim_i2t = F.softmax(topk_sim_i2t, dim=0)[0] # topk_sim_i2t.shape=128
        proba_sim_t2i_top1_idx_i2t = torch.Tensor([0.]).to(proba_top1_sim_i2t.device)#torch.zeros(1).to(proba_top1_sim_i2t.device)
        topk_sim_t2i_top1_idx_i2t, topk_idx_t2i_top1_idx_i2t = sims_matrix_t2i[topk_idx_i2t[0]].topk(k=k_test, dim=0)
        if i in topk_idx_t2i_top1_idx_i2t:
            idx_i_in_topk_idx_t2i_top1_idx_i2t = torch.where(topk_idx_t2i_top1_idx_i2t == i)
            proba_sim_t2i_top1_idx_i2t = F.softmax(topk_sim_t2i_top1_idx_i2t, dim=0)[idx_i_in_topk_idx_t2i_top1_idx_i2t]
        top1_match_coeffi = torch.exp((proba_top1_sim_i2t + proba_sim_t2i_top1_idx_i2t)/2)

        # itm tta with uncertainty
        loss_entropy_topk_gallery = -(F.softmax(score, dim=0) * F.log_softmax(score, dim=0)).sum()
        loss_entropy_uncertainty = top1_match_coeffi * loss_entropy_topk_gallery.mean()
        loss_entropy_uncertainty = loss_entropy_uncertainty / itm_loss_backward_accum_bs
        loss_entropy_uncertainty.backward()
        grad_accum_num += 1
        if grad_accum_num >= itm_loss_backward_accum_bs or i>=end-1:
            optimizer.step()
            optimizer.zero_grad()
            grad_accum_num = 0

        score_matrix_i2t[start+i, topk_idx_i2t] = score + topk_sim_i2t * model.temp

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    # model.to(score_matrix_i2t.device)
    torch.cuda.empty_cache()
    return score_matrix_i2t.cpu().detach().numpy()

def compute_t2i_sim_matrix_adapt_zhh_topk(model, data_loader, optimizer,  **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation itm re-rank:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    logging.info("    text features...")
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []

    model.eval()
    with torch.no_grad():
        for i in range(0, num_text, text_bs):
            print("\r text_batch_i: ", i, end="")
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
        print()

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)

        logging.info("    image features...")
        vit_feats = []
        image_embeds = []
        for i, samples in enumerate(data_loader):
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

        logging.info("    similarity matrix...") # 这里F.normaliza()后的矩阵相乘@ 就是cosine_similarity
        sims_matrix = []
        for image_embed in image_embeds: # 5000,32,256
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010 -> 25010
            sim_i2t = sim_i2t / model.temp
            sims_matrix.append(sim_i2t)
        sims_matrix_i2t = torch.stack(sims_matrix, dim=0) #5000,25010
        sims_matrix_t2i = sims_matrix_i2t.t()

    torch.cuda.empty_cache()

    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_t2i.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_t2i.size(0), start + step)
    itm_loss_backward_accum_bs = 64
    grad_accum_num = 0

    model.train()
    for i, sims_t2i in enumerate(
        metric_logger.log_every(sims_matrix_t2i[start:end], 50, header)
    ): # 遍历每个image与25010个text的sim_matrix
        topk_sim_t2i, topk_idx_t2i = sims_t2i.topk(k=k_test, dim=0) #sims.shape=1,25010 topk_sim.shape=128 topk_idx=top128_idx
        image_inputs = vit_feats[topk_idx_t2i.cpu()].to(model.device) # vit_feats[start + i].shape=677,1408 image_inputs.shape=128,677,1408
        score = model.compute_itm(
            image_inputs=image_inputs, # 将一张图片的vit_feat repeat 128次，对齐topk(k=k_test)个候选text #128,677,1408
            text_ids=text_ids[start + i].repeat(k_test, 1), #topk(k=k_test)个候选text #128,35
            text_atts=text_atts[start + i].repeat(k_test, 1), #topk(k=k_test)个候选text的attention_mask 128,35
        ).float() # score.shape=128

        proba_top1_sim_t2i = F.softmax(topk_sim_t2i, dim=0)[0]
        proba_sim_i2t_top1_idx_t2i = torch.Tensor([0.]).to(proba_top1_sim_t2i.device)
        topk_sim_i2t_top1_idx_t2i, topk_idx_i2t_top1_idx_t2i = sims_matrix_i2t[topk_idx_t2i[0]].topk(k=k_test, dim=0)
        if i in topk_idx_i2t_top1_idx_t2i:
            idx_of_i_in_topk_idx_i2t_top1_idx_t2i = torch.where(topk_idx_i2t_top1_idx_t2i == i)
            proba_sim_i2t_top1_idx_t2i = F.softmax(topk_sim_i2t_top1_idx_t2i, dim=0)[idx_of_i_in_topk_idx_i2t_top1_idx_t2i]

        top1_match_coeffi = torch.exp((proba_top1_sim_t2i + proba_sim_i2t_top1_idx_t2i)/2)

        # itm tta with uncertainty
        # loss_ent = zhh_softmax_entropy(score)#.mean(0)
        loss_entropy_topk_gallery = -(F.softmax(score, dim=0) * F.log_softmax(score, dim=0)).sum()
        # loss_entropy_uncertainty = ( ( torch.exp((1 + sims[topk_idx])/2) - 1 )  * loss_entropy_topk_gallery ).mean()
        # loss_entropy_uncertainty = ( (1 + sims[topk_idx])/2  * loss_entropy_topk_gallery + (1 - (1 + sims[topk_idx])/2) ).mean()
        # loss_entropy_uncertainty = ( (1 + sims[topk_idx])/2  * loss_entropy_topk_gallery ).mean()
        loss_entropy_uncertainty = top1_match_coeffi * loss_entropy_topk_gallery.mean()
        loss_entropy_uncertainty = loss_entropy_uncertainty / itm_loss_backward_accum_bs
        loss_entropy_uncertainty.backward()
        grad_accum_num += 1
        if grad_accum_num >= itm_loss_backward_accum_bs or i>=end-1:
            optimizer.step()
            optimizer.zero_grad()
            grad_accum_num = 0


        score_matrix_t2i[start + i, topk_idx_t2i] = score + topk_sim_t2i * model.temp

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    # model.to(score_matrix_t2i.device)
    torch.cuda.empty_cache()
    return score_matrix_t2i.cpu().detach().numpy()

# Sample selection for image & text inter-top1 matching
@torch.no_grad()
def sample_selection_intertop1(sims_matrix_i2t, sims_matrix_t2i):

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()

    logging.info("    compute topk index of similarity matrix...")
    selected_sample_idx_i2t = []
    selected_sample_idx_t2i = []

    top5_idx_matrix_i2t = torch.full(
        (sims_matrix_i2t.size(0), 5), fill_value=-100, dtype=torch.int64
    ).to(sims_matrix_i2t.device)
    top1_idx_matrix_t2i = torch.full(
        (sims_matrix_t2i.size(0), 1), fill_value=-100, dtype=torch.int64
    ).to(sims_matrix_i2t.device)

    step = sims_matrix_i2t.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_i2t.size(0), start + step)
    for i, sims in enumerate(sims_matrix_i2t[start:end]): #sims.shape=25010
        top5_sim, top5_idx = sims.topk(k=5, dim=0) # 每个image有5个text 故取top5用于sample selection的inter-top1匹配
        top5_idx_matrix_i2t[start + i, ...] = top5_idx

    step = sims_matrix_t2i.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_t2i.size(0), start + step)
    for i, sims in enumerate(sims_matrix_t2i[start:end]): #sims.shape=5000
        top1_sim, top1_idx = sims.topk(k=1, dim=0)
        top1_idx_matrix_t2i[start + i, ...] = top1_idx

    for idx in range(top5_idx_matrix_i2t.size(0)):
        top5_idx = top5_idx_matrix_i2t[idx, ...]
        if idx in top1_idx_matrix_t2i[top5_idx, ...]:
            selected_sample_idx_i2t.append(idx)

    for idx in range(top1_idx_matrix_t2i.size(0)):
        top1_idx = top1_idx_matrix_t2i[idx, ...]
        if idx in top5_idx_matrix_i2t[top1_idx, ...]:
            selected_sample_idx_t2i.append(idx)

    del top5_idx_matrix_i2t, top1_idx_matrix_t2i
    torch.cuda.empty_cache()
    return selected_sample_idx_i2t, selected_sample_idx_t2i

# 计算i2t和t2的sim matrix
# 计算i2t和t2i互相命中recall@1作为sample selection
@torch.no_grad()
def compute_sim_matrix_sample_selction(model, data_loader, optimizer,  **kwargs):
    logging.info("Computing features for sim matrix...")

    model.eval()
    with torch.no_grad():
        logging.info("    text features...")
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []

        for i in range(0, num_text, text_bs):
            print("\r text_sample_number: ", i, end="")
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

        logging.info("    image features...")
        # vit_feats = []
        # image_embeds = []
        # for i, samples in enumerate(data_loader):
        #     print("\r image_sample_number: ", i*len(samples["image"]), end="")
        #     image = samples["image"]

        #     image = image.to(model.device)
        #     image_feat, vit_feat = model.forward_image(image)
        #     image_embed = model.vision_proj(image_feat)
        #     image_embed = F.normalize(image_embed, dim=-1)

        #     vit_feats.append(vit_feat.cpu())
        #     image_embeds.append(image_embed)

        # vit_feats = torch.cat(vit_feats, dim=0)
        # image_embeds = torch.cat(image_embeds, dim=0)

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

        logging.info("    similarity matrix...") # 这里F.normaliza()后的矩阵相乘@ 就是cosine_similarity
        sims_matrix = []
        for image_embed in image_embeds: # 5000,32,256
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010 -> 25010
            sim_i2t = sim_i2t / model.temp
            sims_matrix.append(sim_i2t)
        sims_matrix_i2t = torch.stack(sims_matrix, dim=0) #5000,25010
        sims_matrix_t2i = sims_matrix_i2t.t()

    selected_sample_idx_i2t, selected_sample_idx_t2i = sample_selection_intertop1(sims_matrix_i2t, sims_matrix_t2i)

    return sims_matrix_i2t.detach(), sims_matrix_t2i.detach(), selected_sample_idx_i2t, selected_sample_idx_t2i, vit_feats, text_ids, text_atts

@torch.no_grad()
def compute_i2t_sim_matrix_adapt_zhh_topk_sample_selection(model, data_loader, optimizer, **kwargs):
    k_test = kwargs.pop("k_test")
    selected_sample_idx_i2t = kwargs.pop("selected_sample_idx_i2t")
    sims_matrix_i2t = kwargs.pop("sims_matrix_i2t")
    sims_matrix_t2i = kwargs.pop("sims_matrix_t2i")
    vit_feats = kwargs.pop("vit_feats")
    text_ids = kwargs.pop("text_ids")
    text_atts = kwargs.pop("text_atts")
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation itm re-rank:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()

    model.train()

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(data_loader.dataset.text)), -100.0
    ).to(model.device)

    step = sims_matrix_i2t.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_i2t.size(0), start + step)
    itm_loss_backward_accum_bs = 64
    grad_accum_num = 0
    with torch.enable_grad():
        for i, sims_i2t in enumerate(
            metric_logger.log_every(sims_matrix_i2t[start:end], 100, header)
        ): # 遍历每个image与25010个text的sim_matrix
            topk_sim_i2t, topk_idx_i2t = sims_i2t.topk(k=k_test, dim=0) #sims.shape=25010 topk_sim.shape=128 topk_idx=top128_idx
            image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device) # vit_feats[i].shape=1,677,1408 image_inputs.shape=128,677,1408
            score = model.compute_itm(
                image_inputs=image_inputs, # 将一张图片的vit_feat repeat 128次，对齐topk(k=k_test)个候选text #128,677,1408
                text_ids=text_ids[topk_idx_i2t], #topk(k=k_test)个候选text #128,35
                text_atts=text_atts[topk_idx_i2t], #topk(k=k_test)个候选text的attention_mask 128,35
            ).float() # score.shape=128

            if i in selected_sample_idx_i2t:

                proba_top1_sim_i2t = F.softmax(topk_sim_i2t)[0]
                proba_sim_t2i_top1_idx_i2t = torch.zeros(1).to(proba_top1_sim_i2t.device)
                topk_sim_t2i_top1_idx_i2t, topk_idx_t2i_top1_idx_i2t = sims_matrix_t2i[topk_idx_i2t[0]].topk(k=k_test, dim=0)
                if i in topk_idx_t2i_top1_idx_i2t:
                    idx_i_in_topk_idx_t2i_top1_idx_i2t = torch.where(topk_idx_t2i_top1_idx_i2t == i)
                    proba_sim_t2i_top1_idx_i2t = F.softmax(topk_sim_t2i_top1_idx_i2t)[idx_i_in_topk_idx_t2i_top1_idx_i2t]
                top1_match_coeffi = torch.exp((proba_top1_sim_i2t + proba_sim_t2i_top1_idx_i2t)/2)

                # itm tta with uncertainty
                loss_entropy_topk_gallery = -(F.softmax(score) * F.log_softmax(score)).sum()
                loss_entropy_uncertainty = top1_match_coeffi * loss_entropy_topk_gallery.mean()
                loss_entropy_uncertainty = loss_entropy_uncertainty / itm_loss_backward_accum_bs
                loss_entropy_uncertainty.backward()
                grad_accum_num += 1
                if grad_accum_num >= itm_loss_backward_accum_bs or i>=end-1:
                    optimizer.step()
                    optimizer.zero_grad()
                    grad_accum_num = 0

            score_matrix_i2t[start+i, topk_idx_i2t] = score + topk_sim_i2t

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().detach().numpy()

@torch.no_grad()
def compute_t2i_sim_matrix_adapt_zhh_topk_sample_selection(model, data_loader, optimizer,  **kwargs):
    k_test = kwargs.pop("k_test")
    selected_sample_idx_t2i = kwargs.pop("selected_sample_idx_t2i")
    sims_matrix_i2t = kwargs.pop("sims_matrix_i2t")
    sims_matrix_t2i = kwargs.pop("sims_matrix_t2i")
    vit_feats = kwargs.pop("vit_feats")
    text_ids = kwargs.pop("text_ids")
    text_atts = kwargs.pop("text_atts")
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation itm re-rank:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()

    model.train()

    score_matrix_t2i = torch.full(
        (len(len(data_loader.dataset.text)), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix_t2i.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_t2i.size(0), start + step)
    itm_loss_backward_accum_bs = 64
    grad_accum_num = 0
    with torch.enable_grad():
        for i, sims_t2i in enumerate(
            metric_logger.log_every(sims_matrix_t2i[start:end], 100, header)
        ): # 遍历每个image与25010个text的sim_matrix
            topk_sim_t2i, topk_idx_t2i = sims_t2i.topk(k=k_test, dim=0) #sims.shape=1,25010 topk_sim.shape=128 topk_idx=top128_idx
            image_inputs = vit_feats[topk_idx_t2i.cpu()].to(model.device) # vit_feats[start + i].shape=677,1408 image_inputs.shape=128,677,1408
            score = model.compute_itm(
                image_inputs=image_inputs, # 将一张图片的vit_feat repeat 128次，对齐topk(k=k_test)个候选text #128,677,1408
                text_ids=text_ids[start + i].repeat(k_test, 1), #topk(k=k_test)个候选text #128,35
                text_atts=text_atts[start + i].repeat(k_test, 1), #topk(k=k_test)个候选text的attention_mask 128,35
            ).float() # score.shape=128

            if i in selected_sample_idx_t2i:
                proba_top1_sim_t2i = F.softmax(topk_sim_t2i)[0]
                proba_sim_i2t_top1_idx_t2i = torch.Tensor([0.]).to(proba_top1_sim_t2i.device)
                topk_sim_i2t_top1_idx_t2i, topk_idx_i2t_top1_idx_t2i = sims_matrix_i2t[topk_idx_t2i[0]].topk(k=k_test, dim=0)
                if i in topk_idx_i2t_top1_idx_t2i:
                    idx_of_i_in_topk_idx_i2t_top1_idx_t2i = torch.where(topk_idx_i2t_top1_idx_t2i == i)
                    proba_sim_i2t_top1_idx_t2i = F.softmax(topk_sim_i2t_top1_idx_t2i)[idx_of_i_in_topk_idx_i2t_top1_idx_t2i]

                top1_match_coeffi = torch.exp((proba_top1_sim_t2i + proba_sim_i2t_top1_idx_t2i)/2)

                # itm tta with uncertainty
                # loss_ent = zhh_softmax_entropy(score)#.mean(0)
                loss_entropy_topk_gallery = -(F.softmax(score) * F.log_softmax(score)).sum()
                # loss_entropy_uncertainty = ( ( torch.exp((1 + sims[topk_idx])/2) - 1 )  * loss_entropy_topk_gallery ).mean()
                # loss_entropy_uncertainty = ( (1 + sims[topk_idx])/2  * loss_entropy_topk_gallery + (1 - (1 + sims[topk_idx])/2) ).mean()
                # loss_entropy_uncertainty = ( (1 + sims[topk_idx])/2  * loss_entropy_topk_gallery ).mean()
                loss_entropy_uncertainty = top1_match_coeffi * loss_entropy_topk_gallery.mean()
                loss_entropy_uncertainty = loss_entropy_uncertainty / itm_loss_backward_accum_bs
                loss_entropy_uncertainty.backward()
                grad_accum_num += 1
                if grad_accum_num >= itm_loss_backward_accum_bs or i>=end-1:
                    optimizer.step()
                    optimizer.zero_grad()
                    grad_accum_num = 0

            score_matrix_t2i[start + i, topk_idx_t2i] = score + topk_sim_t2i

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_t2i.cpu().detach().numpy()

def compute_i2t_sim_matrix_adapt_itm(model, data_loader, optimizer, tta_cfg, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "i2t online Evaluation itm adapt:"

    logging.info("i2t online Computing features for evaluation...")
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        logging.info("    text features...")
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
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

        logging.info("    image features...")
        vit_feats = []
        image_embeds = []
        for i, samples in enumerate(data_loader):
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

        logging.info("    similarity matrix...") # 这里F.normaliza()后的矩阵相乘@ 就是cosine_similarity
        sims_matrix = []
        for image_embed in image_embeds: # 5000,32,256
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010 -> 25010
            # sim_i2t = sim_i2t / model.temp
            sims_matrix.append(sim_i2t)
        sims_matrix_i2t = torch.stack(sims_matrix, dim=0) #5000,25010
        sims_matrix_t2i = sims_matrix_i2t.t()

    model.train()

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_i2t.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_i2t.size(0), start + step)
    itm_loss_backward_accum_bs = 64 #1 #64
    grad_accum_num = 0
    for i, sims_i2t in enumerate(
        metric_logger.log_every(sims_matrix_i2t[start:end], 50, header)
    ): # 遍历每个image与25010个text的sim_matrix
        topk_sim_i2t, topk_idx_i2t = sims_i2t.topk(k=k_test, dim=0) #sims.shape=25010 topk_sim.shape=128 topk_idx=top128_idx
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device) # vit_feats[i].shape=1,677,1408 image_inputs.shape=128,677,1408
        score = model.compute_itm(
            image_inputs=image_inputs, #128,677,1408
            text_ids=text_ids[topk_idx_i2t], #128,35
            text_atts=text_atts[topk_idx_i2t], #128,35
        ).float() # score.shape=128

        top1_match_coeffi = torch.ones(1).to(model.device)
        if hasattr(tta_cfg, 'top1_match_coeffi') and tta_cfg.top1_match_coeffi == True:
            proba_top1_sim_i2t = F.softmax(topk_sim_i2t, dim=0)[0]
            proba_sim_t2i_top1_idx_i2t = torch.zeros(1).to(proba_top1_sim_i2t.device)
            topk_sim_t2i_top1_idx_i2t, topk_idx_t2i_top1_idx_i2t = sims_matrix_t2i[topk_idx_i2t[0]].topk(k=k_test, dim=0)
            if i in topk_idx_t2i_top1_idx_i2t:
                idx_i_in_topk_idx_t2i_top1_idx_i2t = torch.where(topk_idx_t2i_top1_idx_i2t == i)
                proba_sim_t2i_top1_idx_i2t = F.softmax(topk_sim_t2i_top1_idx_i2t, dim=0)[idx_i_in_topk_idx_t2i_top1_idx_i2t]
            top1_match_coeffi = torch.exp(1 - (proba_top1_sim_i2t + proba_sim_t2i_top1_idx_i2t)/2 )

        # itm entropy adapt
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

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("i2t online Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().detach().numpy()

def compute_t2i_sim_matrix_adapt_itm(model, data_loader, optimizer, tta_cfg, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "i2t online Evaluation itm adapt:"

    logging.info("i2t online Computing features for evaluation...")
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        logging.info("    text features...")
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
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

        logging.info("    image features...")
        vit_feats = []
        image_embeds = []
        for i, samples in enumerate(data_loader):
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

        logging.info("    similarity matrix...") # 这里F.normaliza()后的矩阵相乘@ 就是cosine_similarity
        sims_matrix = []
        for image_embed in image_embeds: # 5000,32,256
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010 -> 25010
            # sim_i2t = sim_i2t / model.temp
            sims_matrix.append(sim_i2t)
        sims_matrix_i2t = torch.stack(sims_matrix, dim=0) #5000,25010
        sims_matrix_t2i = sims_matrix_i2t.t()

    model.train()

    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_t2i.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_t2i.size(0), start + step)
    itm_loss_backward_accum_bs = 64
    grad_accum_num = 0
    for i, sims_t2i in enumerate(
        metric_logger.log_every(sims_matrix_t2i[start:end], 50, header)
    ): # 遍历每个image与25010个text的sim_matrix
        topk_sim_t2i, topk_idx_t2i = sims_t2i.topk(k=k_test, dim=0) #sims.shape=1,25010 topk_sim.shape=128 topk_idx=top128_idx
        image_inputs = vit_feats[topk_idx_t2i.cpu()].to(model.device) # vit_feats[start + i].shape=677,1408 image_inputs.shape=128,677,1408
        score = model.compute_itm(
            image_inputs=image_inputs, #128,677,1408
            text_ids=text_ids[start + i].repeat(k_test, 1), #128,35
            text_atts=text_atts[start + i].repeat(k_test, 1), #128,35
        ).float() # score.shape=128

        top1_match_coeffi = torch.ones(1).to(model.device)
        if hasattr(tta_cfg, 'top1_match_coeffi') and tta_cfg.top1_match_coeffi == True:
            proba_top1_sim_t2i = F.softmax(topk_sim_t2i, dim=0)[0]
            proba_sim_i2t_top1_idx_t2i = torch.Tensor([0.]).to(proba_top1_sim_t2i.device)
            topk_sim_i2t_top1_idx_t2i, topk_idx_i2t_top1_idx_t2i = sims_matrix_i2t[topk_idx_t2i[0]].topk(k=k_test, dim=0)
            if i in topk_idx_i2t_top1_idx_t2i:
                idx_of_i_in_topk_idx_i2t_top1_idx_t2i = torch.where(topk_idx_i2t_top1_idx_t2i == i)
                proba_sim_i2t_top1_idx_t2i = F.softmax(topk_sim_i2t_top1_idx_t2i, dim=0)[idx_of_i_in_topk_idx_i2t_top1_idx_t2i]
            top1_match_coeffi = torch.exp( 1 - (proba_top1_sim_t2i + proba_sim_i2t_top1_idx_t2i)/2 )

        # itm adapt
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
        score_matrix_t2i[start + i, topk_idx_t2i] = score + topk_sim_t2i

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("i2t online Evaluation time {}".format(total_time_str))

    return score_matrix_t2i.cpu().detach().numpy()

@torch.no_grad()
def compute_i2t_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "i2t offline Evaluation:"

    logging.info("i2t offline Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    logging.info("    text features...")
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
    logging.info("    image features...")
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
    logging.info("    cosine similarity...")
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

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("i2t offline Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), _, sims_matrix.cpu().numpy()

@torch.no_grad()
def compute_t2i_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "t2i offline Evaluation:"

    logging.info("t2i offline Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    logging.info("    text features...")
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
    logging.info("    image features...")
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
    logging.info("    cosine similarity...")
    for image_embed in image_embeds: # 5000,32,256
        sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
        sim_i2t, _ = sim_q2t.max(0) #32,25010
        sims_matrix.append(sim_i2t) # 25010
    sims_matrix = torch.stack(sims_matrix, dim=0) #5000,25010
    sims_matrix = sims_matrix.t()

    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
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
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("t2i offline Evaluation time {}".format(total_time_str))

    return _, score_matrix_t2i.cpu().numpy(), sims_matrix.cpu().numpy()

def compute_i2t_sim_matrix_adapt_itm_ss(model, data_loader, optimizer, tta_cfg, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation online for i2t itm adapt:"

    logging.info("Computing features for i2t online evaluation...")
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        logging.info("    text features...")
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
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
        text_ids = torch.cat(text_ids, dim=0).detach().cpu()
        text_atts = torch.cat(text_atts, dim=0).detach().cpu()

        logging.info("    image features...")
        vit_feats = []
        image_embeds = []
        for i, samples in enumerate(data_loader):
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

        logging.info("    similarity matrix...") # 这里F.normaliza()后的矩阵相乘@ 就是cosine_similarity
        sims_matrix = []
        for image_embed in image_embeds: # 5000,32,256
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010 -> 25010
            # sim_i2t = sim_i2t / model.temp
            sims_matrix.append(sim_i2t)
        sims_matrix_i2t = torch.stack(sims_matrix, dim=0).detach().cpu() #5000,25010
        # sims_matrix_i2t = image_embeds.detach().cpu() @ text_embeds.t().detach().cpu()
        # sims_matrix_i2t = sims_matrix_i2t.max(1).values 
        sims_matrix_t2i = sims_matrix_i2t.t().detach().cpu()
        del sims_matrix
        torch.cuda.empty_cache()

        selected_sample_idx_i2t, selected_sample_idx_t2i = sample_selection_intertop1(sims_matrix_i2t, sims_matrix_t2i.detach().cpu())
        logging.info(f"     after sample selection, number of i2t samples: {len(selected_sample_idx_i2t)}, number of t2i samples: {len(selected_sample_idx_t2i)}")

    model.train()
    model.visual_encoder.to('cpu')
    model.ln_vision.to('cpu')
    model.vision_proj.to('cpu')
    model.text_proj.to('cpu')
    torch.cuda.empty_cache()

    model.query_tokens.to(model.device)
    model.Qformer.to(model.device)
    model.itm_head.to(model.device)
    
    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).detach().cpu()

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_i2t.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_i2t.size(0), start + step)
    itm_loss_backward_accum_bs = 64
    grad_accum_num = 0
    for i, sims_i2t in enumerate(
        metric_logger.log_every(sims_matrix_i2t[start:end], 50, header)
    ): # 遍历每个image与25010个text的sim_matrix
        topk_sim_i2t, topk_idx_i2t = sims_i2t.topk(k=k_test, dim=0) #sims.shape=25010 topk_sim.shape=128 topk_idx=top128_idx
        topk_sim_i2t = topk_sim_i2t.to(model.device)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device) # vit_feats[i].shape=1,677,1408 image_inputs.shape=128,677,1408
        # print(i)
        score = model.compute_itm(
            image_inputs=image_inputs, #128,677,1408
            text_ids=text_ids[topk_idx_i2t].to(model.device), #128,35
            text_atts=text_atts[topk_idx_i2t].to(model.device), #128,35
        ).float() # score.shape=128

        if i in selected_sample_idx_i2t:
            top1_match_coeffi = torch.ones(1).to(model.device)
            if hasattr(tta_cfg, 'top1_match_coeffi') and tta_cfg.top1_match_coeffi == True:
                proba_top1_sim_i2t = F.softmax(topk_sim_i2t, dim=0)[0]
                proba_sim_t2i_top1_idx_i2t = torch.zeros(1).to(proba_top1_sim_i2t.device)
                topk_sim_t2i_top1_idx_i2t, topk_idx_t2i_top1_idx_i2t = sims_matrix_t2i[topk_idx_i2t[0]].topk(k=k_test, dim=0)
                topk_sim_t2i_top1_idx_i2t, topk_idx_t2i_top1_idx_i2t =  topk_sim_t2i_top1_idx_i2t.to(model.device), topk_idx_t2i_top1_idx_i2t.to(model.device)
                if i in topk_idx_t2i_top1_idx_i2t:
                    idx_i_in_topk_idx_t2i_top1_idx_i2t = torch.where(topk_idx_t2i_top1_idx_i2t == i)
                    proba_sim_t2i_top1_idx_i2t = F.softmax(topk_sim_t2i_top1_idx_i2t, dim=0)[idx_i_in_topk_idx_t2i_top1_idx_i2t]
                top1_match_coeffi = torch.exp(1 - (proba_top1_sim_i2t + proba_sim_t2i_top1_idx_i2t)/2 )

            # itm entropy adapt
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
        score_matrix_i2t[start+i, topk_idx_i2t.detach().cpu()] = score.detach().cpu() + topk_sim_i2t.detach().cpu()
        torch.cuda.empty_cache()

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("i2t online Evaluation time {}".format(total_time_str))

    model.visual_encoder.to(model.device)
    model.ln_vision.to(model.device)
    model.vision_proj.to(model.device)
    model.text_proj.to(model.device)
    # model.to(model.device)
    return score_matrix_i2t.cpu().detach().numpy()

def compute_t2i_sim_matrix_adapt_itm_ss(model, data_loader, optimizer, tta_cfg, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation online for t2i itm adapt:"

    logging.info("Computing features for t2i online evaluation...")
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        logging.info("    text features...")
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
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
        text_ids = torch.cat(text_ids, dim=0).detach().cpu()
        text_atts = torch.cat(text_atts, dim=0).detach().cpu()

        logging.info("    image features...")
        vit_feats = []
        image_embeds = []
        for i, samples in enumerate(data_loader):
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

        logging.info("    similarity matrix...") # 这里F.normaliza()后的矩阵相乘@ 就是cosine_similarity
        sims_matrix = []
        for image_embed in image_embeds: # 5000,32,256
            sim_q2t = image_embed @ text_embeds.t() # image_embed.shape=32,256 text_embeds.shape=25010,256
            sim_i2t, _ = sim_q2t.max(0) #32,25010 -> 25010
            # sim_i2t = sim_i2t / model.temp
            sims_matrix.append(sim_i2t)
        sims_matrix_i2t = torch.stack(sims_matrix, dim=0).detach().cpu() #5000,25010
        sims_matrix_t2i = sims_matrix_i2t.t().detach().cpu()
        
        del sims_matrix
        torch.cuda.empty_cache()

        selected_sample_idx_i2t, selected_sample_idx_t2i = sample_selection_intertop1(sims_matrix_i2t, sims_matrix_t2i)
        logging.info(f"after sample selection, number of i2t samples: {len(selected_sample_idx_i2t)}, number of t2i samples: {len(selected_sample_idx_t2i)}")


    model.train()
    model.visual_encoder.to('cpu')
    model.ln_vision.to('cpu')
    model.vision_proj.to('cpu')
    model.text_proj.to('cpu')
    torch.cuda.empty_cache()

    model.query_tokens.to(model.device)
    model.Qformer.to(model.device)
    model.itm_head.to(model.device)

    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).detach().cpu()

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix_t2i.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_t2i.size(0), start + step)
    itm_loss_backward_accum_bs = 64
    grad_accum_num = 0
    for i, sims_t2i in enumerate(
        metric_logger.log_every(sims_matrix_t2i[start:end], 50, header)
    ): # 遍历每个image与25010个text的sim_matrix
        topk_sim_t2i, topk_idx_t2i = sims_t2i.topk(k=k_test, dim=0) #sims.shape=1,25010 topk_sim.shape=128 topk_idx=top128_idx
        topk_sim_t2i = topk_sim_t2i.to(model.device)
        image_inputs = vit_feats[topk_idx_t2i.cpu()].to(model.device) # vit_feats[start + i].shape=677,1408 image_inputs.shape=128,677,1408
        score = model.compute_itm(
            image_inputs=image_inputs, #128,677,1408
            text_ids=text_ids[start + i].repeat(k_test, 1).to(model.device), #128,35
            text_atts=text_atts[start + i].repeat(k_test, 1).to(model.device), #128,35
        ).float() # score.shape=128

        if i in selected_sample_idx_t2i:
            top1_match_coeffi = torch.ones(1).to(model.device)
            if hasattr(tta_cfg, 'top1_match_coeffi') and tta_cfg.top1_match_coeffi == True:
                proba_top1_sim_t2i = F.softmax(topk_sim_t2i, dim=0)[0]
                proba_sim_i2t_top1_idx_t2i = torch.Tensor([0.]).to(proba_top1_sim_t2i.device)
                topk_sim_i2t_top1_idx_t2i, topk_idx_i2t_top1_idx_t2i = sims_matrix_i2t[topk_idx_t2i[0]].topk(k=k_test, dim=0)
                topk_sim_i2t_top1_idx_t2i, topk_idx_i2t_top1_idx_t2i =  topk_sim_i2t_top1_idx_t2i.to(model.device), topk_idx_i2t_top1_idx_t2i.to(model.device)
                if i in topk_idx_i2t_top1_idx_t2i:
                    idx_of_i_in_topk_idx_i2t_top1_idx_t2i = torch.where(topk_idx_i2t_top1_idx_t2i == i)
                    proba_sim_i2t_top1_idx_t2i = F.softmax(topk_sim_i2t_top1_idx_t2i, dim=0)[idx_of_i_in_topk_idx_i2t_top1_idx_t2i]
                top1_match_coeffi = torch.exp( 1 - (proba_top1_sim_t2i + proba_sim_i2t_top1_idx_t2i)/2 )

            # itm adapt
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
        score_matrix_t2i[start + i, topk_idx_t2i.detach().cpu()] = score.detach().cpu() + topk_sim_t2i.detach().cpu()

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("i2t online Evaluation time {}".format(total_time_str))

    model.visual_encoder.to(model.device)
    model.ln_vision.to(model.device)
    model.vision_proj.to(model.device)
    model.text_proj.to(model.device)
    # model.to(model.device)
    return score_matrix_t2i.cpu().detach().numpy()

