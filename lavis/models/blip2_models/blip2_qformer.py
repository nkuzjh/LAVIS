"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    compute_sim_matrix_worerank,
    disabled_train,
    compute_i2t_sim_matrix_adapt,
    compute_t2i_sim_matrix_adapt,
    compute_i2t_sim_matrix_adapt_worerank,
    compute_t2i_sim_matrix_adapt_worerank,
    compute_i2t_sim_matrix_adapt_zhh,
    compute_t2i_sim_matrix_adapt_zhh,
    compute_i2t_sim_matrix_adapt_zhh_topk,
    compute_t2i_sim_matrix_adapt_zhh_topk,
    compute_sim_matrix_sample_selction,
    compute_i2t_sim_matrix_adapt_zhh_topk_sample_selection,
    compute_t2i_sim_matrix_adapt_zhh_topk_sample_selection,

    compute_i2t_sim_matrix_adapt_itm,
    compute_t2i_sim_matrix_adapt_itm,
    compute_i2t_sim_matrix,
    compute_t2i_sim_matrix,
    compute_i2t_sim_matrix_adapt_itm_ss,
    compute_t2i_sim_matrix_adapt_itm_ss,
    compute_i2t_sim_matrix_adapt_itm_sigmoid,
    compute_t2i_sim_matrix_adapt_itm_sigmoid,
)
from lavis.models.blip2_models.blip2_speed_up import compute_i2t_sim_matrix_adapt_itm as compute_i2t_sim_matrix_adapt_itm_vislog
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def forward(self, samples):
        image = samples["image"] #14,3,364,364
        text = samples["text_input"] #list.len=14

        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(
            image_feats#14,32,256
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim] #14,32,256
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim] #14，256

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1) #14,32,256 * 14,256 = 14, 14, 32
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1) # 14,14
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1) # 14,256 * 14,256,32 = 14, 14, 32
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1) # 14, 14
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]#均值5左右，最大值11左右

        try:
            rank = dist.get_rank() # 仅用于pretrain时的分布式训练时local_batch中retrieval similarity matrix的labels分配，见下述torch.linespace方法；单机单卡训练将rank置为0即可
        except:
            rank = 0
        bs = image.size(0) #14
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )# list of range(0,14,1)

        if "image_id" in samples.keys(): #coco retrieval finetuning
            image_ids = torch.as_tensor(samples["image_id"], dtype=torch.double).view(-1,1).to(sim_t2i.device) # 14,1
            image_ids_all = concat_all_gather(image_ids) #14,1
            pos_idx = torch.eq(image_ids, image_ids_all.t()).float() # 14,14 bs*bs的对角线为1的mask矩阵（单位矩阵I）
            sim_targets = pos_idx / pos_idx.sum(1,keepdim=True) # 单卡训练同上，bs*bs的单位矩阵
            sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1)# 手动实现的label_smoothing,用于一个image_id可能对应多个text_id的情况，即多标签分类问题的cross_entropy

            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean()
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
            loss_itc = (loss_t2i+loss_i2t)/2
        else:
            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)# 多分类cross_entropy
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2

        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids) #14,32 #self.max_txt_len=32
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask) #14,32 #self.max_txt_len=32
        image_embeds_world = all_gather_with_grad(image_embeds)#14,677,1024
        with torch.no_grad():
            if "image_id" in samples.keys():
                mask = torch.eq(image_ids, image_ids_all.t())
                sim_t2i.masked_fill_(mask, -10000)# mask=1的位置替换为-10000，mask=0的位置数值不变
                sim_i2t.masked_fill_(mask, -10000)
            else:
                sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)

            weights_t2i = F.softmax(sim_t2i, dim=1)# 14，14 按照cos_similarity的概率 进行负样本抽样；除了单位矩阵对角线的image-text样本对，其他位置的cos_sim越大可以视为困难负样本
            weights_i2t = F.softmax(sim_i2t, dim=1)# 14，14

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item() # 多项式概率抽样，抽样次数为1,返回索引
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)#每个text样本抽样一个image负样本 14,677,1024

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0) # 14,32 #self.max_txt_len=32
        text_atts_neg = torch.stack(text_atts_neg, dim=0) # 14,32 #self.max_txt_len=32

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg #42,32
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        ) #42,32

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1) #self.query_tokens=1,32,768 -> query_tokens_itm.shape=42,32,768
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )# 42,32
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1) #42,64

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos #42.677.1024
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )#42,677

        output_itm = self.Qformer.bert(
            text_ids_all, #42,32#这里设计的42=14*3即bs*3分别代表了image_pos-text_pos，image_neg-text_pos， image_pos-text_neg三种样本对
            query_embeds=query_tokens_itm,#42,32,768
            attention_mask=attention_mask_all,#42,64 #64=32+32 #attention_mask_all=torch.cat([query_atts_itm, text_atts_all]) query_atts_itm为图片的query_tokens的mask，且全为1，text_atts_all为文本的attention_mask，根据文本长度进行mask；二者长度都为32，但一个指的是图片的query_tokens数量，一个指的是token的self.max_txt_len；Qformer的输入本身就是query和text concatenate后作为token sequence input，初始的hidden state则由image_embeds作为input；
            encoder_hidden_states=image_embeds_all,#42,677,1024
            encoder_attention_mask=image_atts_all,#42,677
            return_dict=True,
        )#output_itm.last_hidden_state.shape=42,64,768

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]#output_itm.last_hidden_state=42,64,768;query_tokens_itm.size(1)=32;vl_embeddings=42,32,768 #只要query_tokens_itm部分的 hidden state sequence token 输出
        vl_output = self.itm_head(vl_embeddings) # vl_embeddings=42,32,768 -> vl_output=42,32,2 #单类别的分类任务改为多分类(2个类别)任务
        logits = vl_output.mean(dim=1)#42,32,2 -> 42,2 #32个query_token的平均值作为ITM的预测结果 #与ITC使用max(query_token)作为预测结果不同

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,#42 #对应了image_pos-text_pos，image_neg-text_pos， image_pos-text_neg三种样本对
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)#

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()#14,32
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id#把input_ids的第一位从cls_token_id换成bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )# pad_token_id换成-100

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(#query_tokens=14,32,768
            image.device
        )#14,32
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)#14,64 #64=query32+text_max_len32
        lm_output = self.Qformer( #self.Qformer=lavis.models.blip2_models.Qformer.BertLMHeadModel #self.Qformer.bert=lavis.models.blip2_models.Qformer.BertModel
            decoder_input_ids,#14,32
            attention_mask=attention_mask,#14,64
            past_key_values=query_output.past_key_values, #12*2的二维tuple * 14,12,32,64的tensor #Qformer.bert 输入query和image_embed 输出query_output 其中query_output.last_hidden_state=14,32,768经过vision_proj后成为image_feat 而query_output.past_key_values作为frozen LLM Decoder的初始hidden_state直接输入self.Qformer.forward
            # past_key_values 缓存了模型在 先前时间步 计算过的键（Key）和值（Value）张量，使得在生成下一个 token 时：
            # 避免重复计算：直接复用已缓存的键值，减少冗余计算。
            # 实现高效的自回归生成：只需计算当前步的注意力，而不需要重新处理整个历史序列。
            # past_key_values 是一个 嵌套的元组或列表，其结构如下：
            # 每个元素对应模型的一个解码器层（Decoder Layer）。
            # 每个层包含两个张量：(past_key, past_value)，形状通常为：
            # (batch_size, num_heads, seq_len, head_dim)
            # 其中 seq_len 是已生成序列的长度。
            return_dict=True,
            labels=labels,# 14,32
        )#lm_output.keys()=['loss', 'logits'] #lm_output.logits.shape=14,32,30523 #self.tokenizer.vocab_size=30522

        loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts): # shape = #128,677,1408 #128,35 #128,35
        # self.visual_encoder.to('cpu')
        # self.ln_vision.to('cpu')
        # self.vision_proj.to('cpu')
        # self.text_proj.to('cpu')
        # torch.cuda.empty_cache()

        # self.query_tokens.to(image_inputs.device)
        # self.Qformer.to(image_inputs.device)
        # self.itm_head.to(image_inputs.device)
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )#  image_inputs.shape=128,677,1408 -> image_atts.shape=128,677
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1).to(
            image_inputs.device
        ) #self.query_tokens.shape=1,32,768 -> query_tokens.shap=128,32,768
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )# 128,32
        attention_mask = torch.cat([query_atts, text_atts], dim=1).to(
            image_inputs.device
        ) #128,67
        output_itm = self.Qformer.bert(
            text_ids, #128,35
            query_embeds=query_tokens, #128,32,768
            attention_mask=attention_mask, #128,67
            encoder_hidden_states=image_inputs, #128,677,1408
            encoder_attention_mask=image_atts, #128,677
            return_dict=True,
        )# output_itm.last_hidden_state=128,67,768
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]#128,32,768
        itm_logit = self.itm_head(vl_embeddings)# 128,32,2
        itm_logit = itm_logit[:, :, 1].mean(dim=1) #itm_logit[:, :, 1].shape=128,32 #itm_logit[:, :, 1].mean(dim=1).shape=128
        
        # self.visual_encoder.to(image_inputs.device)
        # self.ln_vision.to(image_inputs.device)
        # self.vision_proj.to(image_inputs.device)
        # self.text_proj.to(image_inputs.device)
        # self.to(image_inputs.device)
        # torch.cuda.empty_cache()
        # self.query_tokens.to(image_inputs.device)
        # self.Qformer.to(image_inputs.device)
        # self.itm_head.to(image_inputs.device)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g") # 'clip_L'
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg, wo_rerank):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        # return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
        if wo_rerank:
            return compute_sim_matrix_worerank(model=self, data_loader=data_loader, k_test=k_test)
        else:
            return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

    def compute_i2t_sim_matrix_adapt(self, data_loader, task_cfg, optimizer, wo_rerank):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        if wo_rerank:
            return compute_i2t_sim_matrix_adapt_worerank(model=self, data_loader=data_loader, optimizer=optimizer,  k_test=k_test)
        else:
            return compute_i2t_sim_matrix_adapt(model=self, data_loader=data_loader, optimizer=optimizer,  k_test=k_test)

    def compute_t2i_sim_matrix_adapt(self, data_loader, task_cfg, optimizer, wo_rerank):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        if wo_rerank:
            return compute_t2i_sim_matrix_adapt_worerank(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test)
        else:
            return compute_t2i_sim_matrix_adapt(model=self, data_loader=data_loader, optimizer=optimizer,  k_test=k_test)

    def compute_i2t_sim_matrix_adapt_zhh(self, data_loader, task_cfg, optimizer, wo_rerank):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_i2t_sim_matrix_adapt_zhh(model=self, data_loader=data_loader, optimizer=optimizer,  k_test=k_test)
        # if wo_rerank:
        #     return compute_i2t_sim_matrix_adapt_zhh_worerank(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test)
        # else:
        #     return compute_i2t_sim_matrix_adapt_zhh(model=self, data_loader=data_loader, optimizer=optimizer,  k_test=k_test)

    def compute_t2i_sim_matrix_adapt_zhh(self, data_loader, task_cfg, optimizer, wo_rerank):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_t2i_sim_matrix_adapt_zhh(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test)
        # if wo_rerank:
        #     return compute_t2i_sim_matrix_adapt_zhh_worerank(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test)
        # else:
        #     return compute_t2i_sim_matrix_adapt_zhh(model=self, data_loader=data_loader, optimizer=optimizer,  k_test=k_test)

    def compute_i2t_sim_matrix_adapt_zhh_topk(self, data_loader, task_cfg, optimizer, tta_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        score_i2t = compute_i2t_sim_matrix_adapt_zhh_topk(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test)
        return score_i2t

    def compute_t2i_sim_matrix_adapt_zhh_topk(self, data_loader, task_cfg, optimizer, tta_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        score_t2i = compute_t2i_sim_matrix_adapt_zhh_topk(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test)
        return score_t2i

    def compute_sim_matrix_sample_selction(self, data_loader, task_cfg, optimizer, tta_cfg):
        k_test = task_cfg.k_test

        return compute_sim_matrix_sample_selction(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test)

    def compute_i2t_sim_matrix_adapt_zhh_topk_sample_selection(self, data_loader, task_cfg, optimizer, tta_cfg, selected_sample_idx_i2t, sims_matrix_i2t, sims_matrix_t2i, vit_feats, text_ids, text_atts):
        k_test = task_cfg.k_test

        score_t2i = compute_i2t_sim_matrix_adapt_zhh_topk_sample_selection(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test, selected_sample_idx_i2t=selected_sample_idx_i2t, sims_matrix_i2t=sims_matrix_i2t, sims_matrix_t2i=sims_matrix_t2i, vit_feats=vit_feats, text_ids=text_ids, text_atts=text_atts)
        return score_t2i

    def compute_t2i_sim_matrix_adapt_zhh_topk_sample_selection(self, data_loader, task_cfg, optimizer, tta_cfg, selected_sample_idx_t2i):
        k_test = task_cfg.k_test

        score_t2i = compute_t2i_sim_matrix_adapt_zhh_topk_sample_selection(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test, selected_sample_idx_t2i=selected_sample_idx_t2i, sims_matrix_i2t=sims_matrix_i2t, sims_matrix_t2i=sims_matrix_t2i, vit_feats=vit_feats, text_ids=text_ids, text_atts=text_atts)
        return score_t2i

    def compute_i2t_sim_matrix_adapt_itm(self, data_loader, task_cfg, optimizer, tta_cfg, epoch=None):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        if hasattr(tta_cfg, "debug_visual") and tta_cfg.debug_visual == True:
            score_i2t = compute_i2t_sim_matrix_adapt_itm_vislog(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test, tta_cfg=tta_cfg, epoch=epoch)
        else:
            score_i2t = compute_i2t_sim_matrix_adapt_itm(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test, tta_cfg=tta_cfg)
        return score_i2t

    def compute_t2i_sim_matrix_adapt_itm(self, data_loader, task_cfg, optimizer, tta_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        score_t2i = compute_t2i_sim_matrix_adapt_itm(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test, tta_cfg=tta_cfg)
        return score_t2i

    def compute_i2t_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        score_i2t = compute_i2t_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
        return score_i2t

    def compute_t2i_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        score_t2i = compute_t2i_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
        return score_t2i


    def compute_i2t_sim_matrix_adapt_itm_ss(self, data_loader, task_cfg, optimizer, tta_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        score_i2t = compute_i2t_sim_matrix_adapt_itm_ss(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test, tta_cfg=tta_cfg)
        return score_i2t

    def compute_t2i_sim_matrix_adapt_itm_ss(self, data_loader, task_cfg, optimizer, tta_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        score_t2i = compute_t2i_sim_matrix_adapt_itm_ss(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test, tta_cfg=tta_cfg)
        return score_t2i
    
    def compute_i2t_sim_matrix_adapt_itm_sigmoid(self, data_loader, task_cfg, optimizer, tta_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        score_i2t = compute_i2t_sim_matrix_adapt_itm_sigmoid(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test, tta_cfg=tta_cfg)
        return score_i2t

    def compute_t2i_sim_matrix_adapt_itm_sigmoid(self, data_loader, task_cfg, optimizer, tta_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        score_t2i = compute_t2i_sim_matrix_adapt_itm_sigmoid(model=self, data_loader=data_loader, optimizer=optimizer, k_test=k_test, tta_cfg=tta_cfg)
        return score_t2i