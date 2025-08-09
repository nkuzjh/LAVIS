import json
from typing import Iterable
import pandas as pd
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class TTA_I2T_Dataset(Dataset):
    def __init__(
        self, tta_cfg, sims_matrix_all, sims_matrix, sims_idxs, labels, recall_types, recall_types_2, vit_feats, text_ids, text_atts, tta_coeffis,  proba_top1_sim_list, proba_sim_at_top1_idx_list
    ):
        self.tta_cfg = tta_cfg
        # self.sims_matrix_all = sims_matrix_all
        self.sims_matrix = sims_matrix
        self.sims_idxs = sims_idxs
        self.labels = labels
        # self.recall_types = recall_types
        self.recall_types_2 = recall_types_2
        self.vit_feats = vit_feats
        self.text_ids = text_ids
        self.text_atts = text_atts
        self.tta_coeffis = tta_coeffis
        self.proba_top1_sim_list = proba_top1_sim_list
        self.proba_sim_at_top1_idx_list = proba_sim_at_top1_idx_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # print("TTA_I2T_Dataset __getitem__ start")
        ### sims_matrix_all
        # sims_matrix_all = self.sims_matrix_all[index]  # shape: (1, k_test)
        # print("TTA_I2T_Dataset __getitem__ start1")
        ### cos_sims_matrix
        sims = self.sims_matrix[index]
        idxs = self.sims_idxs[index]
        # print("TTA_I2T_Dataset __getitem__ start2")
        ### vis feature
        image_inputs = self.vit_feats[index].unsqueeze(0).repeat(1, self.tta_cfg.k_tta, 1, 1)# vit_feats[i].shape=1,677,1408; image_inputs.shape=1,k_tta,677,1408
        image_inputs = image_inputs.reshape(-1, image_inputs.size(-2), image_inputs.size(-1)) # 1*k_tta,677,1408
        # print("TTA_I2T_Dataset __getitem__ start3")
        ### txt feature
        text_ids_inputs = self.text_ids[idxs]
        text_ids_inputs = text_ids_inputs.reshape(-1, text_ids_inputs.size(-1)) # 1*k_tta,35
        text_atts_inputs = self.text_atts[idxs]
        text_atts_inputs = text_atts_inputs.reshape(-1, text_atts_inputs.size(-1)) # 1*k_tta,35
        # print("TTA_I2T_Dataset __getitem__ start4")
        ### entropy coeffi
        tta_coeffi = self.tta_coeffis[index] # shape=1
        if getattr(self.tta_cfg, "coeffi_exp_temper_is_learnable", False) == True:
            tta_coeffi_proba1 = self.proba_top1_sim_list[index]
            tta_coeffi_proba2 = self.proba_sim_at_top1_idx_list[index]
        else:
            tta_coeffi_proba1 = torch.ones(1)
            tta_coeffi_proba2 = torch.ones(1)
        ### label
        label = self.labels[index]
        # recall_type = self.recall_types[index] # 字符串
        recall_type_2 = self.recall_types_2[index]
        # print("TTA_I2T_Dataset __getitem__ start5")

        return {
            "index": torch.Tensor([index]),
            # "sims_all": sims_matrix_all,
            "sims": sims,
            "idxs": idxs,
            "image_inputs": image_inputs,
            "text_ids": text_ids_inputs,
            "text_atts": text_atts_inputs,
            "tta_coeffi": tta_coeffi,
            "tta_coeffi_proba1": tta_coeffi_proba1,
            "tta_coeffi_proba2": tta_coeffi_proba2,
            "label": label,
            # "recall_type": recall_type,
            "recall_type_2": recall_type_2,
        }

    def collater(self, batch):
        """
        Args:
            batch: list of dicts with keys:
                'idx', 'image_inputs', 'text_ids', 'text_atts',
                'tta_coeffi', 'label', 'recall_type', 'recall_type_2'

        Returns:
            dict of batched tensors and lists
        """
        collated = {}
        # 特殊处理 label (变长list of int)，保留为 list
        collated['label'] = [item['label'] for item in batch]

        # 特殊处理 recall_type, recall_type_2（字符串），保留为（字符串），保留为 list
        # collated['recall_type'] = [item['recall_type'] for item in batch]
        collated['recall_type_2'] = [item['recall_type_2'] for item in batch]

        # 其他张量字段使用 default_collate
        for key in batch[0]:
            if key in [ 'recall_type_2', 'label']:
                continue  # 已经处理过了

            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = default_collate(values)
                except:
                    print()
                    print(key)
                    print(values)
            else:
                collated[key] = values  # 如果是非 tensor 字段（如 label 是 int）

        return collated

class TTA_T2I_Dataset(Dataset):
    def __init__(
        self, tta_cfg, sims_matrix, sims_idxs, labels, recall_types, vit_feats, text_ids, text_atts, tta_coeffis
    ):
        self.tta_cfg = tta_cfg
        self.sims_matrix = sims_matrix
        self.sims_idxs = sims_idxs
        self.labels = labels
        self.recall_types = recall_types
        self.vit_feats = vit_feats
        self.text_ids = text_ids
        self.text_atts = text_atts
        self.tta_coeffis = tta_coeffis

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ### cos_sims_matrix
        sims = self.sims_matrix[index]
        idxs = self.sims_idxs[index]
        ### txt feature
        text_ids_inputs = self.text_ids[index] # 35
        text_ids_inputs = text_ids_inputs.unsqueeze(0).repeat(self.tta_cfg.k_tta, 1) # 1*k_tta, 35
        text_atts_inputs = self.text_atts[index]
        text_atts_inputs = text_atts_inputs.unsqueeze(0).repeat(self.tta_cfg.k_tta, 1) # 1*k_tta, 35
        ### vis feature
        image_inputs = self.vit_feats[idxs]
        image_inputs = image_inputs.reshape(-1, image_inputs.size(-2), image_inputs.size(-1))# vit_feats[i].shape=k_tta,677,1408; image_inputs.shape=1*k_tta,677,1408
        ### entropy coeffi
        tta_coeffi = self.tta_coeffis[index] # shape=1
        ### label
        label = self.labels[index]
        recall_type = self.recall_types[index] # 字符串

        return {
            "index": torch.Tensor([index]),
            "sims": sims,
            "idxs": idxs,
            "image_inputs": image_inputs,
            "text_ids": text_ids_inputs,
            "text_atts": text_atts_inputs,
            "tta_coeffi": tta_coeffi,
            "label": label,
            "recall_type": recall_type,
        }

    def collater(self, batch):
        """
        Args:
            batch: list of dicts with keys:
                'idx', 'image_inputs', 'text_ids', 'text_atts',
                'tta_coeffi', 'label', 'recall_type'

        Returns:
            dict of batched tensors and lists
        """
        collated = {}
        # 特殊处理 label (变长list of int)，保留为 list
        collated['label'] = [item['label'] for item in batch]

        # 特殊处理 recall_type（字符串），保留为 list
        collated['recall_type'] = [item['recall_type'] for item in batch]

        # 其他张量字段使用 default_collate
        for key in batch[0]:
            if key in ['recall_type', 'label']:
                continue  # 已经处理过了

            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = default_collate(values)
                except:
                    print()
                    print(key)
                    print(values)
            else:
                collated[key] = values  # 如果是非 tensor 字段（如 label 是 int）

        return collated



class TTA_I2T_Shards_Dataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        # 只扫描目录下的 .pt 文件，不加载任何数据
        self.file_list = sorted(
            [f for f in os.listdir(dataset_dir) if f.endswith('.pt')],
            key=lambda x: int(x.split('.')[0])
        )
        logging.info(f"    find {len(self.file_list)} samples in {dataset_dir}")
        assert len(self.file_list) > 0, f"No .pt files found in {dataset_dir}"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # 按需加载单个样本
        data_path = os.path.join(self.dataset_dir, self.file_list[index])
        data = torch.load(data_path)

        # 重构原始 __getitem__ 的逻辑
        k_tta = data['tta_cfg_k_tta']

        # vis feature
        image_inputs = data['vit_feats'].unsqueeze(0).repeat(1, k_tta, 1, 1)
        image_inputs = image_inputs.reshape(-1, image_inputs.size(-2), image_inputs.size(-1))

        # txt feature
        text_ids_inputs = data['text_ids'].reshape(-1, data['text_ids'].size(-1))
        text_atts_inputs = data['text_atts'].reshape(-1, data['text_atts'].size(-1))

        # coeffi
        tta_coeffi = data['tta_coeffi']
        tta_coeffi_proba1 = data['proba_top1_sim']
        tta_coeffi_proba2 = data['proba_sim_at_top1_idx']

        # label & meta
        label = data['label']
        recall_type_2 = data['recall_type_2']

        return {
            "index": torch.tensor([index]),
            "sims": data['sims'],
            "idxs": data['idxs'],
            "image_inputs": image_inputs,
            "text_ids": text_ids_inputs,
            "text_atts": text_atts_inputs,
            "tta_coeffi": tta_coeffi,
            "tta_coeffi_proba1": tta_coeffi_proba1,
            "tta_coeffi_proba2": tta_coeffi_proba2,
            "label": label,
            "recall_type_2": recall_type_2,
        }
    
    def collater(self, batch):
        """
        Args:
            batch: list of dicts with keys:
                'idx', 'image_inputs', 'text_ids', 'text_atts',
                'tta_coeffi', 'label', 'recall_type', 'recall_type_2'

        Returns:
            dict of batched tensors and lists
        """
        collated = {}
        # 特殊处理 label (变长list of int)，保留为 list
        collated['label'] = [item['label'] for item in batch]

        # 特殊处理 recall_type, recall_type_2（字符串），保留为（字符串），保留为 list
        # collated['recall_type'] = [item['recall_type'] for item in batch]
        collated['recall_type_2'] = [item['recall_type_2'] for item in batch]

        # 其他张量字段使用 default_collate
        for key in batch[0]:
            if key in [ 'recall_type_2', 'label']:
                continue  # 已经处理过了

            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = default_collate(values)
                except:
                    print()
                    print(key)
                    print(values)
            else:
                collated[key] = values  # 如果是非 tensor 字段（如 label 是 int）

        return collated



import logging
import numpy as np
from lavis.common.dist_utils import get_rank, get_world_size
from lavis.datasets.datasets.dataloader_utils import PrefetchLoader
from torch.utils.data import DataLoader, DistributedSampler
from lavis.datasets.datasets.retrieval_datasets import RetrievalEvalDataset
from lavis.processors.blip_processors import BlipImageEvalProcessor, BlipCaptionProcessor
from tta.utils import calculate_recall, sample_neg_idxs, compute_tta_coeffis, find_inter_top1_sample_selection


def create_eval_dataset(cfg):

    vis_processor = BlipImageEvalProcessor(image_size=364) # official baseline: cls.from_config(cfg=cfg.datasets.coco_retrieval.vis_processor.eval)
    text_processor = BlipCaptionProcessor() # official baseline: cls.from_config(cfg=cfg.datasets.coco_retrieval.text_processor.eval)

    ann_paths = ['/data/jiahao/coco/annotations/coco_karpathy_test.json']
    vis_path = '/data/jiahao/coco/images/' # dataset_builder(datasets_config[name]).build_info.data_type.vis_info.storage

    dataset = RetrievalEvalDataset(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
            )

    return dataset

def create_eval_dataloader(cfg, dataset):   

    if cfg.run_cfg.distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=False,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.run_cfg.batch_size_eval,
        num_workers=cfg.run_cfg.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=False,
        collate_fn=getattr(dataset, "collater", None),
        drop_last=False,
    )
    dataloader = PrefetchLoader(dataloader)

    return dataloader


def save_dataset_shards(dataset_dir, tta_cfg, sims_matrix, sims_idxs, labels, recall_types_2, vit_feats, text_ids, text_atts, tta_coeffis, proba_top1_sim_list, proba_sim_at_top1_idx_list):
    """
    将每个样本的数据保存为单独的 .pt 文件
    文件名: {index}.pt
    """
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"saving dataset shards to {dataset_dir}")

    for idx in tqdm(range(len(labels))):
        data = {
            'tta_cfg_k_tta': tta_cfg.k_tta,  # 只保存必要的配置
            'sims': torch.tensor(sims_matrix[idx]),
            'idxs': torch.tensor(sims_idxs[idx]),
            'vit_feats': torch.tensor(vit_feats[idx]) if isinstance(vit_feats, (list, tuple)) else vit_feats[idx].clone(),
            'text_ids': torch.tensor(text_ids[sims_idxs[idx]]),
            'text_atts': torch.tensor(text_atts[sims_idxs[idx]]),
            'tta_coeffi': torch.tensor(tta_coeffis[idx]),
            'proba_top1_sim': torch.tensor(proba_top1_sim_list[idx]) if isinstance(proba_top1_sim_list, (list, tuple)) else torch.ones(1),
            'proba_sim_at_top1_idx': torch.tensor(proba_sim_at_top1_idx_list[idx]) if isinstance(proba_sim_at_top1_idx_list, (list, tuple)) else torch.ones(1),
            'label': labels[idx],
            'recall_type_2': recall_types_2[idx]
        }
        torch.save(data, os.path.join(dataset_dir, f"{idx}.pt"))

def preprocess_tta_dataset(cfg, labels, sims_matrix_i2t, vit_feats, text_ids, text_atts):
    logging.info(f"preprocess_tta_dataset  start")
    tta_cfg = cfg.config.tta
   
    ## 获取metric标签用于可视化
    k_test = cfg.run_cfg.k_test
    top128_sims, top128_idxs = sims_matrix_i2t.topk(k=k_test, dim=1)
    (recall1, recall5, recall10), recall_types_2 = calculate_recall(top128_idxs, labels)

    ## sampling stretegy
    logging.info(f"pos/neg sampling ...")
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
    logging.info("number of sample after pos/neg sampling: {}".format(sampled_sims_matrix_i2t.size()))
    
    logging.info(f"tta_coeffis & score_temper ...")
    ## tta coeffis
    if tta_cfg.top1_match_coeffi == True:
        if getattr(tta_cfg, "coeffi_exp_temper_is_learnable", False) == True:
            tta_coeffis, proba_top1_sim_list, proba_sim_at_top1_idx_list = compute_tta_coeffis(sims_matrix_i2t, sims_matrix_i2t.t(), k_test, tta_cfg.coeffi_i2t_temper, tta_cfg.coeffi_t2i_temper, tta_cfg)
        else:
            tta_coeffis = compute_tta_coeffis(sims_matrix_i2t, sims_matrix_i2t.t(), k_test, tta_cfg.coeffi_i2t_temper, tta_cfg.coeffi_t2i_temper)
    else:
        tta_coeffis = [ torch.ones(1) for _ in range(sims_matrix_i2t.size(0)) ]
    ## score temperature
    score_temper_ = tta_cfg.score_temper if hasattr(tta_cfg, "score_temper") else 1.0

    logging.info(f"sample_selection ...")
    ## sample selection stretegy
    if tta_cfg.sample_selection == "top1":
        ss_idxs = find_inter_top1_sample_selection(sims_matrix_i2t, sims_matrix_i2t.t()) # 找到i2t和t2i互为top1的样本索引
        # logging.info("    ss_idxs")
        # 只保留top1 sample selection的样本
        sims_matrix_i2t = sims_matrix_i2t[ss_idxs]
        # vit_feats = vit_feats.to(model.device)
        vit_feats = vit_feats[ss_idxs]
        # logging.info("    vit_feats")
        # text_ids = text_ids[ss_idxs]
        # text_atts = text_atts[ss_idxs]
        labels = [labels[i] for i in ss_idxs]
        # recall_types = [recall_types[i] for i in ss_idxs]
        recall_types_2 = [recall_types_2[i] for i in ss_idxs]
        # logging.info("    labels recall_types")
        # top1_sims, top1_idxs = top1_sims[ss_idxs], top1_idxs[ss_idxs]
        # neg_sims, neg_idxs = neg_sims[ss_idxs], neg_idxs[ss_idxs]
        sampled_sims_matrix_i2t = sampled_sims_matrix_i2t[ss_idxs]
        sampled_sims_idx_i2t = sampled_sims_idx_i2t[ss_idxs]
        # logging.info("    sampled_sims_matrix_i2t")
        tta_coeffis = [tta_coeffis[i] for i in ss_idxs]
        # logging.info("    tta_coeffis")
        # score_matrix_i2t_ = score_matrix_i2t[ss_idxs].cpu()
        # logging.info("    score_matrix_i2t_")
        # scores_mat_ = scores_mat[ss_idxs].cpu()
        # logging.info("    scores_mat")
        # labels_mat_2_ = labels_mat_[ss_idxs]
    else:
        ss_idxs = torch.arange(0, sims_matrix_i2t.size(0)) # 全部样本
        # score_matrix_i2t_ = score_matrix_i2t[ss_idxs].cpu()
        # scores_mat_ = scores_mat[ss_idxs].cpu()
        # labels_mat_2_ = labels_mat_[ss_idxs]
    logging.info("number of sample after sample_selection: {}".format(len(ss_idxs)))

    ## tta_dataset & tta_dataloader
    logging.info(f"tta_dataset ...")
    # if getattr(tta_cfg, "coeffi_exp_temper_is_learnable", False) == True:
    #     tta_dataset = TTA_I2T_Dataset(
    #         tta_cfg,
    #         None, sampled_sims_matrix_i2t, sampled_sims_idx_i2t,
    #         labels, None, recall_types_2,
    #         vit_feats, text_ids, text_atts,
    #         tta_coeffis, proba_top1_sim_list, proba_sim_at_top1_idx_list
    #     )
    # else:
    #     tta_dataset = TTA_I2T_Dataset(
    #         tta_cfg,
    #         None, sampled_sims_matrix_i2t, sampled_sims_idx_i2t,
    #         labels, None, recall_types_2,
    #         vit_feats, text_ids, text_atts,
    #         tta_coeffis, None, None
    #     )
    ## to avoid cpu oom, split np.array into separate files
    save_dataset_shards(
        dataset_dir=tta_cfg.shards_dataset_dir,
        tta_cfg=tta_cfg,
        sims_matrix=sampled_sims_matrix_i2t,
        sims_idxs=sampled_sims_idx_i2t,
        labels=labels,
        recall_types_2=recall_types_2,
        vit_feats=vit_feats,
        text_ids=text_ids,
        text_atts=text_atts,
        tta_coeffis=tta_coeffis,
        proba_top1_sim_list=proba_top1_sim_list if getattr(tta_cfg, "coeffi_exp_temper_is_learnable", False) else None,
        proba_sim_at_top1_idx_list=proba_sim_at_top1_idx_list if getattr(tta_cfg, "coeffi_exp_temper_is_learnable", False)  else None
    )
    logging.info(f"preprocess_tta_dataset  end")

def create_tta_dataset(cfg):
    logging.info(f"create_tta_dataset  start")

    tta_cfg = cfg.config.tta
    tta_dataset = TTA_I2T_Shards_Dataset(tta_cfg.shards_dataset_dir)
    logging.info("number of tta_dataset: {}".format(len(tta_dataset)))

    logging.info(f"create_tta_dataset  end")
    return tta_dataset

def create_tta_dataloader(cfg, dataset):
    logging.info(f"create_tta_dataloader  start")

    if cfg.run_cfg.distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.tta_cfg.tta_bs,
        num_workers=cfg.run_cfg.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=getattr(dataset, "collater", None),
        drop_last=True,
    )
    dataloader = PrefetchLoader(dataloader) # 预加载batch数据，加速数据通信
    # dataloader = IterLoader(dataloader, use_distributed=cfg.run_cfg.distributed) # IterLoader同时有以下功能： 1.DDP设置sample.set_epoch(epoch)用于分布式sampler打乱数据 2.转化为无限iter(dataloader)
    logging.info("number of tta_dataloader: {}".format(len(dataloader)))

    logging.info(f"create_tta_dataloader  end")
    return dataloader, sampler