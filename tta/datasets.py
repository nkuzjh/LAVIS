import json
from typing import Iterable
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

class TTA_I2T_Dataset(Dataset):
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
        image_inputs = self.vit_feats[idxs].reshape(-1, image_inputs.size(-2), image_inputs.size(-1))# vit_feats[i].shape=k_tta,677,1408; image_inputs.shape=1*k_tta,677,1408
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
