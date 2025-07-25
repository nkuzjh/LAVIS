import json
from typing import Iterable
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

class TTA_I2T_Dataset(Dataset):
    def __init__(
        self, tta_cfg, sims_matrix_all, sims_matrix, sims_idxs, labels, recall_types, recall_types_2, vit_feats, text_ids, text_atts, tta_coeffis
    ):
        self.tta_cfg = tta_cfg
        self.sims_matrix_all = sims_matrix_all
        self.sims_matrix = sims_matrix
        self.sims_idxs = sims_idxs
        self.labels = labels
        self.recall_types = recall_types
        self.recall_types_2 = recall_types_2
        self.vit_feats = vit_feats
        self.text_ids = text_ids
        self.text_atts = text_atts
        self.tta_coeffis = tta_coeffis

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ### sims_matrix_all
        sims_matrix_all = self.sims_matrix_all[index]  # shape: (1, k_test)
        ### cos_sims_matrix
        sims = self.sims_matrix[index]
        idxs = self.sims_idxs[index]
        ### vis feature
        image_inputs = self.vit_feats[index].unsqueeze(0).repeat(1, self.tta_cfg.k_tta, 1, 1)# vit_feats[i].shape=1,677,1408; image_inputs.shape=1,k_tta,677,1408
        image_inputs = image_inputs.reshape(-1, image_inputs.size(-2), image_inputs.size(-1)) # 1*k_tta,677,1408
        ### txt feature
        text_ids_inputs = self.text_ids[idxs]
        text_ids_inputs = text_ids_inputs.reshape(-1, text_ids_inputs.size(-1)) # 1*k_tta,35
        text_atts_inputs = self.text_atts[idxs]
        text_atts_inputs = text_atts_inputs.reshape(-1, text_atts_inputs.size(-1)) # 1*k_tta,35
        ### entropy coeffi
        tta_coeffi = self.tta_coeffis[index] # shape=1
        ### label
        label = self.labels[index]
        recall_type = self.recall_types[index] # 字符串
        recall_type_2 = self.recall_types_2[index]

        return {
            "index": torch.Tensor([index]),
            "sims_all": sims_matrix_all,
            "sims": sims,
            "idxs": idxs,
            "image_inputs": image_inputs,
            "text_ids": text_ids_inputs,
            "text_atts": text_atts_inputs,
            "tta_coeffi": tta_coeffi,
            "label": label,
            "recall_type": recall_type,
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
        collated['recall_type'] = [item['recall_type'] for item in batch]
        collated['recall_type_2'] = [item['recall_type_2'] for item in batch]

        # 其他张量字段使用 default_collate
        for key in batch[0]:
            if key in ['recall_type', 'recall_type_2', 'label']:
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



def create_eval_dataset(cfg):
    from lavis.datasets.datasets.retrieval_datasets import RetrievalEvalDataset
    from lavis.processors.blip_processors import BlipImageEvalProcessor, BlipCaptionProcessor

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




import logging
import numpy as np

def create_tta_dataset(cfg):

    tta_cfg = cfg.confg.tta

    logging.info(f"all itc features  end")
    sims_matrix_i2t = torch.from_numpy(np.load("debugs/debug_sim_matrix_i2t.npy"))
    sims_matrix_t2i = torch.from_numpy(np.load("debugs/debug_sim_matrix_t2i.npy"))
    image_embeds = torch.from_numpy(np.load("debugs/debug_image_embeds.npy"))
    vit_feats = torch.from_numpy(np.load("/data/jiahao/blip2_embeddings/debug_vit_feats.npy"))
    text_embeds = torch.from_numpy(np.load("debugs/debug_text_embeds.npy"))
    text_ids = torch.from_numpy(np.load("debugs/debug_text_ids.npy"))
    text_atts = torch.from_numpy(np.load("debugs/debug_text_atts.npy"))
    logging.info(f"all itc features  end")

    ## tta_dataset & tta_dataloader
    tta_dataset = TTA_I2T_Dataset(
        tta_cfg,
        sims_matrix_i2t, sampled_sims_matrix_i2t, sampled_sims_idx_i2t,
        labels, recall_types, recall_types_2,
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

    return dataset