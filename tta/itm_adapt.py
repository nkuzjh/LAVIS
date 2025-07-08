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
)

class ITM_ADAPT(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, data_loader, task_cfg, tta_cfg):
        if self.episodic:
            self.reset()

        # for _ in range(self.steps):
        if 1:
            if tta_cfg.name == 'itm_adapt' or tta_cfg.name == 'visenc_itm_adapt' or tta_cfg.name == 'all_itm_adapt':
                outputs = forward_and_itm_adapt(self, self.optimizer, data_loader, task_cfg, tta_cfg)
            elif tta_cfg.name == 'itm_adapt_ss':
                outputs = forward_and_itm_adapt_ss(self, self.optimizer, data_loader, task_cfg, tta_cfg)
            elif tta_cfg.name == 'itm_adapt_sigmoid':
                outputs = forward_and_itm_adapt_sigmoid(self, self.optimizer, data_loader, task_cfg, tta_cfg)
            elif tta_cfg.name == 'itm_adapt_v1':
                outputs = forward_and_itm_adapt_v1(self, self.optimizer, data_loader, task_cfg, tta_cfg)
            elif tta_cfg.name == 'itc_adapt_v1':
                outputs = forward_and_itc_adapt_v1(self, self.optimizer, data_loader, task_cfg, tta_cfg)
            elif tta_cfg.name == 'itm_adapt_v2':
                outputs = forward_and_itm_adapt_v2(self, self.optimizer, data_loader, task_cfg, tta_cfg)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


def configure_model_blip2(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
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

def collect_params_blip2(model):
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

def configure_visenc_model_blip2(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
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
    for m in model.visual_encoder.modules(): # tta更新 visual_encoder 参数；freeze others: query_tokens/temp(erature)/image_proj/text_proj
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def collect_visenc_params_blip2(model):
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
    for nm, m in model.visual_encoder.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_all_model_blip2(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules(): # 针对blip2模型结构，仅tta更新Qformer参数；freeze visual_encoder/query_tokens/temp(erature)/image_proj/text_proj/itm_head的参数；
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def collect_all_params_blip2(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

@torch.no_grad()
def report_metrics(scores_i2t=None, scores_t2i=None, txt2img=None, img2txt=None, prefix_info=""):

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
        os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
    ) as f:
        f.write("\n")
        f.write(prefix_info + " " + json.dumps(eval_result) + "\n")
    return eval_result

@torch.enable_grad()
def forward_and_itm_adapt(tta_model, optimizer, dataloader, task_cfg, tta_cfg):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    score_i2t, score_t2i, score_i2t_offline, score_t2i_offline = None, None, None, None

    if hasattr(tta_cfg, "offline_multi_epochs") and tta_cfg.online == False:
        offline_multi_epochs = tta_cfg.offline_multi_epochs
    else:
        offline_multi_epochs = 1

    # it2
    if "tta_task" not in tta_cfg.keys() or tta_cfg.tta_task == "i2t":
        logging.info("start i2t task for tta")
        if tta_cfg.zero_shot_eval:
            score_i2t_zeroshot, _, _ = tta_model.model.compute_i2t_sim_matrix(dataloader, task_cfg=task_cfg)
            logging.info("report i2t metrics, zero-shot: \r\n")
            logging.info(
                report_metrics(
                    scores_i2t=score_i2t_zeroshot,
                    scores_t2i=None,
                    txt2img=dataloader.dataset.txt2img,
                    img2txt=dataloader.dataset.img2txt,
                    prefix_info="report i2t metrics, zero-shot: ",
                )
            )

        for tta_epoch in range(offline_multi_epochs):
            logging.info(f"start tta epoch {tta_epoch} for i2t task")
            if hasattr(tta_cfg, "debug_visual") and tta_cfg.debug_visual:
                score_i2t = tta_model.model.compute_i2t_sim_matrix_adapt_itm(dataloader, task_cfg, optimizer, tta_cfg, tta_epoch)
            else:
                score_i2t = tta_model.model.compute_i2t_sim_matrix_adapt_itm(dataloader, task_cfg, optimizer, tta_cfg)
            logging.info("report i2t metrics online, at epoch %d :", tta_epoch)
            logging.info(
                report_metrics(
                    scores_i2t=score_i2t,
                    scores_t2i=None,
                    txt2img=dataloader.dataset.txt2img,
                    img2txt=dataloader.dataset.img2txt,
                    prefix_info=f"report i2t metrics online, at epoch {tta_epoch} :"
                )
            )
            torch.cuda.empty_cache()

            if tta_cfg.online == False:
                score_i2t_offline, _, _ = tta_model.model.compute_i2t_sim_matrix(dataloader, task_cfg=task_cfg)
                logging.info("report i2t metrics offline, at epoch %d :\r\n", tta_epoch)
                logging.info(
                    report_metrics(
                        scores_i2t=score_i2t_offline,
                        scores_t2i=None,
                        txt2img=dataloader.dataset.txt2img,
                        img2txt=dataloader.dataset.img2txt,
                        prefix_info=f"report i2t metrics offline, at epoch {tta_epoch} :"
                    )
                )

            # torch.save(tta_model.model.state_dict(), os.path.join(registry.get_path("output_dir"), f"i2t_tta_model_model_epoch_{tta_epoch}.pth"))
            # logging.info(f"save i2t tta model at epoch {tta_epoch} to {os.path.join(registry.get_path('output_dir'), f'i2t_tta_model_model_epoch_{tta_epoch}.pth')}")
            torch.cuda.empty_cache()

    # # reset model to original state before t2i task
    tta_model.reset()

    if "tta_task" not in tta_cfg.keys() or tta_cfg.tta_task == "t2i":
        logging.info("start t2i task for tta")
        if tta_cfg.zero_shot_eval:
            _, score_t2i_zeroshot, _ = tta_model.model.compute_t2i_sim_matrix(dataloader, task_cfg=task_cfg)
            logging.info("report t2i metrics, zero-shot :\r\n")
            logging.info(
                report_metrics(
                    scores_i2t=None,
                    scores_t2i=score_t2i_zeroshot,
                    txt2img=dataloader.dataset.txt2img,
                    img2txt=dataloader.dataset.img2txt,
                    prefix_info=f"report t2i metrics, zero-shot :"
                )
            )
        # t2i
        for tta_epoch in range(offline_multi_epochs):
            logging.info(f"start tta epoch {tta_epoch} for t2i task")
            score_t2i = tta_model.model.compute_t2i_sim_matrix_adapt_itm(dataloader, task_cfg, optimizer, tta_cfg)
            logging.info("report t2i metrics online, at epoch %d :", tta_epoch)
            logging.info(
                report_metrics(
                    scores_i2t=None,
                    scores_t2i=score_t2i,
                    txt2img=dataloader.dataset.txt2img,
                    img2txt=dataloader.dataset.img2txt,
                    prefix_info=f"report t2i metrics online, at epoch {tta_epoch} :"
                )
            )
            torch.cuda.empty_cache()

            if tta_cfg.online == False:
                _, score_t2i_offline, _ = tta_model.model.compute_t2i_sim_matrix(dataloader, task_cfg=task_cfg)
                logging.info("report t2i metrics offline, at epoch %d :", tta_epoch)
                logging.info(
                    report_metrics(
                        scores_i2t=None,
                        scores_t2i=score_t2i_offline,
                        txt2img=dataloader.dataset.txt2img,
                        img2txt=dataloader.dataset.img2txt,
                        prefix_info=f"report t2i metrics offline, at epoch {tta_epoch} :"
                    )
                )

            # torch.save(tta_model.model.state_dict(), os.path.join(registry.get_path("output_dir"), f"t2i_tta_model_model_epoch_{tta_epoch}.pth"))
            # logging.info(f"save t2i tta model at epoch {tta_epoch} to {os.path.join(registry.get_path('output_dir'), f't2i_tta_model_model_epoch_{tta_epoch}.pth')}")
            torch.cuda.empty_cache()

    if tta_cfg.online == False:
        return score_i2t, score_t2i, score_i2t_offline, score_t2i_offline
    else:
        return score_i2t, score_t2i

@torch.enable_grad()
def forward_and_itm_adapt_ss(tta_model, optimizer, dataloader, task_cfg, tta_cfg):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    score_i2t, score_t2i, score_i2t_offline, score_t2i_offline = None, None, None, None

    if hasattr(tta_cfg, "offline_multi_epochs") and tta_cfg.online == False:
        offline_multi_epochs = tta_cfg.offline_multi_epochs
    else:
        offline_multi_epochs = 1

    # it2
    if "tta_task" not in tta_cfg.keys() or tta_cfg.tta_task == "i2t":
        logging.info("start i2t task for tta")
        if tta_cfg.zero_shot_eval:
            score_i2t_zeroshot, _, _ = tta_model.model.compute_i2t_sim_matrix(dataloader, task_cfg=task_cfg)
            logging.info("report i2t metrics, zero-shot: \r\n")
            logging.info(
                report_metrics(
                    scores_i2t=score_i2t_zeroshot,
                    scores_t2i=None,
                    txt2img=dataloader.dataset.txt2img,
                    img2txt=dataloader.dataset.img2txt,
                    prefix_info="report i2t metrics, zero-shot: ",
                )
            )

        for tta_epoch in range(offline_multi_epochs):
            logging.info(f"start tta epoch {tta_epoch} for i2t task")
            score_i2t = tta_model.model.compute_i2t_sim_matrix_adapt_itm_ss(dataloader, task_cfg, optimizer, tta_cfg)
            logging.info("report i2t metrics online, at epoch %d :", tta_epoch)
            logging.info(
                report_metrics(
                    scores_i2t=score_i2t,
                    scores_t2i=None,
                    txt2img=dataloader.dataset.txt2img,
                    img2txt=dataloader.dataset.img2txt,
                    prefix_info=f"report i2t metrics online, at epoch {tta_epoch} :"
                )
            )
            torch.cuda.empty_cache()

            if tta_cfg.online == False:
                score_i2t_offline, _, _ = tta_model.model.compute_i2t_sim_matrix(dataloader, task_cfg=task_cfg)
                logging.info("report i2t metrics offline, at epoch %d :\r\n", tta_epoch)
                logging.info(
                    report_metrics(
                        scores_i2t=score_i2t_offline,
                        scores_t2i=None,
                        txt2img=dataloader.dataset.txt2img,
                        img2txt=dataloader.dataset.img2txt,
                        prefix_info=f"report i2t metrics offline, at epoch {tta_epoch} :"
                    )
                )

            # torch.save(tta_model.model.state_dict(), os.path.join(registry.get_path("output_dir"), f"i2t_tta_model_model_epoch_{tta_epoch}.pth"))
            # logging.info(f"save i2t tta model at epoch {tta_epoch} to {os.path.join(registry.get_path('output_dir'), f'i2t_tta_model_model_epoch_{tta_epoch}.pth')}")
            torch.cuda.empty_cache()

    # # reset model to original state before t2i task
    tta_model.reset()

    if "tta_task" not in tta_cfg.keys() or tta_cfg.tta_task == "t2i":
        logging.info("start t2i task for tta")
        if tta_cfg.zero_shot_eval:
            _, score_t2i_zeroshot, _ = tta_model.model.compute_t2i_sim_matrix(dataloader, task_cfg=task_cfg)
            logging.info("report t2i metrics, zero-shot :\r\n")
            logging.info(
                report_metrics(
                    scores_i2t=None,
                    scores_t2i=score_t2i_zeroshot,
                    txt2img=dataloader.dataset.txt2img,
                    img2txt=dataloader.dataset.img2txt,
                    prefix_info=f"report t2i metrics, zero-shot :"
                )
            )
        # t2i
        for tta_epoch in range(offline_multi_epochs):
            logging.info(f"start tta epoch {tta_epoch} for t2i task")
            score_t2i = tta_model.model.compute_t2i_sim_matrix_adapt_itm_ss(dataloader, task_cfg, optimizer, tta_cfg)
            logging.info("report t2i metrics online, at epoch %d :", tta_epoch)
            logging.info(
                report_metrics(
                    scores_i2t=None,
                    scores_t2i=score_t2i,
                    txt2img=dataloader.dataset.txt2img,
                    img2txt=dataloader.dataset.img2txt,
                    prefix_info=f"report t2i metrics online, at epoch {tta_epoch} :"
                )
            )
            torch.cuda.empty_cache()

            if tta_cfg.online == False:
                _, score_t2i_offline, _ = tta_model.model.compute_t2i_sim_matrix(dataloader, task_cfg=task_cfg)
                logging.info("report t2i metrics offline, at epoch %d :", tta_epoch)
                logging.info(
                    report_metrics(
                        scores_i2t=None,
                        scores_t2i=score_t2i_offline,
                        txt2img=dataloader.dataset.txt2img,
                        img2txt=dataloader.dataset.img2txt,
                        prefix_info=f"report t2i metrics offline, at epoch {tta_epoch} :"
                    )
                )

            # torch.save(tta_model.model.state_dict(), os.path.join(registry.get_path("output_dir"), f"t2i_tta_model_model_epoch_{tta_epoch}.pth"))
            # logging.info(f"save t2i tta model at epoch {tta_epoch} to {os.path.join(registry.get_path('output_dir'), f't2i_tta_model_model_epoch_{tta_epoch}.pth')}")
            torch.cuda.empty_cache()

    if tta_cfg.online == False:
        return score_i2t, score_t2i, score_i2t_offline, score_t2i_offline
    else:
        return score_i2t, score_t2i

@torch.enable_grad()
def forward_and_itm_adapt_sigmoid(tta_model, optimizer, dataloader, task_cfg, tta_cfg):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    score_i2t, score_t2i, score_i2t_offline, score_t2i_offline = None, None, None, None
    if tta_cfg.zero_shot_eval:
        score_i2t_zeroshot, _, _ = tta_model.model.compute_i2t_sim_matrix(dataloader, task_cfg=task_cfg)
        logging.info("report i2t zero-shot metrics : \r\n")
        logging.info(
            report_metrics(
                scores_i2t=score_i2t_zeroshot,
                scores_t2i=None,
                txt2img=dataloader.dataset.txt2img,
                img2txt=dataloader.dataset.img2txt,
                prefix_info="report i2t zero-shot metrics : ",
            )
        )
        np.save("debug1_score_i2t_zeroshot_1.npy",score_i2t_zeroshot)

        score_i2t_zeroshot, score_t2i_zeroshot, _ = tta_model.model.compute_sim_matrix(dataloader, task_cfg=task_cfg, wo_rerank=tta_cfg.wo_rerank)
        logging.info("report zero-shot metrics with origin function : \r\n")
        logging.info(
            report_metrics(
                scores_i2t=score_i2t_zeroshot,
                scores_t2i=score_t2i_zeroshot,
                txt2img=dataloader.dataset.txt2img,
                img2txt=dataloader.dataset.img2txt,
                prefix_info="report zero-shot metrics with origin function : ",
            )
        )
        np.save("debug1_score_i2t_zeroshot.npy",score_i2t_zeroshot)
        np.save("debug1_score_t2i_zeroshot.npy",score_t2i_zeroshot)

        _, score_t2i_zeroshot, _ = tta_model.model.compute_t2i_sim_matrix(dataloader, task_cfg=task_cfg)
        logging.info("report t2i zero-shot metrics : \r\n")
        logging.info(
            report_metrics(
                scores_i2t=None,
                scores_t2i=score_t2i_zeroshot,
                txt2img=dataloader.dataset.txt2img,
                img2txt=dataloader.dataset.img2txt,
                prefix_info="report t2i zero-shot metrics : ",
            )
        )

        np.save("debug1_score_t2i_zeroshot_1.npy",score_t2i_zeroshot)
    if hasattr(tta_cfg, "offline_multi_epochs") and tta_cfg.online == False:
        offline_multi_epochs = tta_cfg.offline_multi_epochs
    else:
        offline_multi_epochs = 1

    # it2
    if "tta_task" not in tta_cfg.keys() or tta_cfg.tta_task == "i2t":
        logging.info("start i2t task for tta")
        for tta_epoch in range(offline_multi_epochs):
            logging.info(f"start tta epoch {tta_epoch} for i2t task")
            score_i2t = tta_model.model.compute_i2t_sim_matrix_adapt_itm_sigmoid(dataloader, task_cfg, optimizer, tta_cfg)
            logging.info("report i2t metrics online, at epoch %d :", tta_epoch)
            logging.info(
                report_metrics(
                    scores_i2t=score_i2t,
                    scores_t2i=None,
                    txt2img=dataloader.dataset.txt2img,
                    img2txt=dataloader.dataset.img2txt,
                    prefix_info=f"report i2t metrics online, at epoch {tta_epoch} :"
                )
            )
            torch.cuda.empty_cache()

            if tta_cfg.online == False:
                score_i2t_offline, _, _ = tta_model.model.compute_i2t_sim_matrix(dataloader, task_cfg=task_cfg)
                logging.info("report i2t metrics offline, at epoch %d :\r\n", tta_epoch)
                logging.info(
                    report_metrics(
                        scores_i2t=score_i2t_offline,
                        scores_t2i=None,
                        txt2img=dataloader.dataset.txt2img,
                        img2txt=dataloader.dataset.img2txt,
                        prefix_info=f"report i2t metrics offline, at epoch {tta_epoch} :"
                    )
                )

            # torch.save(tta_model.model.state_dict(), os.path.join(registry.get_path("output_dir"), f"i2t_tta_model_model_epoch_{tta_epoch}.pth"))
            # logging.info(f"save i2t tta model at epoch {tta_epoch} to {os.path.join(registry.get_path('output_dir'), f'i2t_tta_model_model_epoch_{tta_epoch}.pth')}")
            torch.cuda.empty_cache()

    # # reset model to original state before t2i task
    tta_model.reset()
    if "tta_task" not in tta_cfg.keys() or tta_cfg.tta_task == "t2i":
        logging.info("start t2i task for tta")
        # t2i
        for tta_epoch in range(offline_multi_epochs):
            logging.info(f"start tta epoch {tta_epoch} for t2i task")
            score_t2i = tta_model.model.compute_t2i_sim_matrix_adapt_itm_sigmoid(dataloader, task_cfg, optimizer, tta_cfg)
            logging.info("report t2i metrics online, at epoch %d :", tta_epoch)
            logging.info(
                report_metrics(
                    scores_i2t=None,
                    scores_t2i=score_t2i,
                    txt2img=dataloader.dataset.txt2img,
                    img2txt=dataloader.dataset.img2txt,
                    prefix_info=f"report t2i metrics online, at epoch {tta_epoch} :"
                )
            )
            torch.cuda.empty_cache()

            if tta_cfg.online == False:
                _, score_t2i_offline, _ = tta_model.model.compute_t2i_sim_matrix(dataloader, task_cfg=task_cfg)
                logging.info("report t2i metrics offline, at epoch %d :", tta_epoch)
                logging.info(
                    report_metrics(
                        scores_i2t=None,
                        scores_t2i=score_t2i_offline,
                        txt2img=dataloader.dataset.txt2img,
                        img2txt=dataloader.dataset.img2txt,
                        prefix_info=f"report t2i metrics offline, at epoch {tta_epoch} :"
                    )
                )

            # torch.save(tta_model.model.state_dict(), os.path.join(registry.get_path("output_dir"), f"t2i_tta_model_model_epoch_{tta_epoch}.pth"))
            # logging.info(f"save t2i tta model at epoch {tta_epoch} to {os.path.join(registry.get_path('output_dir'), f't2i_tta_model_model_epoch_{tta_epoch}.pth')}")
            torch.cuda.empty_cache()

    if tta_cfg.online == False:
        return score_i2t, score_t2i, score_i2t_offline, score_t2i_offline
    else:
        return score_i2t, score_t2i


def forward_and_itm_adapt_v1(tta_model, optimizer, dataloader, task_cfg, tta_cfg):
    score_i2t, score_t2i = None, None

    logging.info("compute cosine similarity matrix")
    sim_matrix_i2t, sim_matrix_t2i, image_embeds, vit_feats, text_embeds, text_ids, text_atts = compute_embeds(tta_model.model, dataloader, task_cfg, tta_cfg)

    ## i2t tta
    if tta_cfg.tta_task == "i2t":
        if tta_cfg.zero_shot_eval:
            logging.info("compute i2t itm score, zero-shot")
            score_i2t_zeroshot= compute_i2t_itm_score(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts)
            result = report_metrics(scores_i2t=score_i2t_zeroshot, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics, zero-shot :")
            logging.info(result)
        ## multi epochs
        for tta_epoch in range(tta_cfg.offline_multi_epochs):
            logging.info(f"start i2t tta epoch {tta_epoch}")
            logging.info("adapt i2t itm score online, epoch %d :", tta_epoch)
            score_i2t = adapt_i2t_itm_score(tta_model.model, dataloader, task_cfg, optimizer, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts, tta_epoch)
            results = report_metrics(scores_i2t=score_i2t, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics online, epoch {tta_epoch} :")
            logging.info(results)
            # torch.save(tta_model.model.state_dict(), os.path.join(registry.get_path("output_dir"), f"i2t_tta_model_model_epoch_{tta_epoch}.pth"))
            # logging.info(f"save i2t tta model at epoch {tta_epoch} to {os.path.join(registry.get_path('output_dir'), f'i2t_tta_model_model_epoch_{tta_epoch}.pth')}")

            logging.info("compute i2t itm score offline, epoch %d :", tta_epoch)
            score_i2t= compute_i2t_itm_score(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts, tta_epoch)
            results = report_metrics(scores_i2t=score_i2t, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics offline, epoch {tta_epoch} :")
            logging.info(results)

    ## reset model to original state before t2i task
    tta_model.reset()
    ## t2i tta
    if tta_cfg.tta_task == "t2i":
        if tta_cfg.zero_shot_eval:
            logging.info("compute t2i itm score, zero-shot")
            score_t2i_zeroshot= compute_t2i_itm_score(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_t2i, vit_feats, text_ids, text_atts)
            result = report_metrics(scores_i2t=None, scores_t2i=score_t2i_zeroshot, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i metrics, zero-shot :")
            logging.info(result)
        ## multi epochs
        for tta_epoch in range(tta_cfg.offline_multi_epochs):
            logging.info(f"start t2i tta epoch {tta_epoch}")
            logging.info("adapt t2i itm score online, epoch %d :", tta_epoch)
            score_t2i = adapt_t2i_itm_score(tta_model.model, dataloader, task_cfg, optimizer, tta_cfg, sim_matrix_t2i, vit_feats, text_ids, text_atts, tta_epoch)
            results = report_metrics(scores_i2t=None, scores_t2i=score_t2i, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i metrics online, epoch {tta_epoch} :")
            logging.info(results)
            # torch.save(tta_model.model.state_dict(), os.path.join(registry.get_path("output_dir"), f"t2i_tta_model_model_epoch_{tta_epoch}.pth"))
            # logging.info(f"save t2i tta model at epoch {tta_epoch} to {os.path.join(registry.get_path('output_dir'), f't2i_tta_model_model_epoch_{tta_epoch}.pth')}")
            logging.info("compute t2i itm score offline, epoch %d :", tta_epoch)
            score_t2i= compute_t2i_itm_score(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_t2i, vit_feats, text_ids, text_atts, tta_epoch)
            results = report_metrics(scores_i2t=None, scores_t2i=score_t2i, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i metrics offline, epoch {tta_epoch} :")
            logging.info(results)

    return score_i2t, score_t2i


def forward_and_itc_adapt_v1(tta_model, optimizer, dataloader, task_cfg, tta_cfg):

    score_i2t, score_t2i = None, None
    logging.info("compute cosine similarity matrix, zero-shot")
    sim_matrix_i2t_zs, sim_matrix_t2i_zs, image_embeds_zs, vit_feats_zs, text_embeds_zs, text_ids_zs, text_atts_zs = compute_embeds(tta_model.model, dataloader, task_cfg, tta_cfg)

    ## i2t tta
    if tta_cfg.tta_task == "i2t":
        if tta_cfg.zero_shot_eval:
            logging.info("compute i2t itm score, zero-shot")
            score_i2t_zeroshot= compute_i2t_itm_score(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_i2t_zs, vit_feats_zs, text_ids_zs, text_atts_zs)
            result = report_metrics(scores_i2t=score_i2t_zeroshot, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics, zero-shot :")
            logging.info(result)
        ## multi epochs
        for tta_epoch in range(tta_cfg.offline_multi_epochs):
            logging.info(f"start i2t tta epoch {tta_epoch}")

            logging.info("adapt cosine similarity matrix, online")
            sim_matrix_i2t, image_embeds, vit_feats, text_embeds, text_ids, text_atts = adapt_i2t_itc_sim(tta_model.model, dataloader, task_cfg, optimizer, tta_cfg)
            logging.info("compute i2t itm score online, epoch %d :", tta_epoch)
            score_i2t= compute_i2t_itm_score(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts)
            results = report_metrics(scores_i2t=score_i2t, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics online, at epoch {tta_epoch} :")
            logging.info(results)
            # torch.save(tta_model.model.state_dict(), os.path.join(registry.get_path("output_dir"), f"i2t_tta_model_model_epoch_{tta_epoch}.pth"))
            # logging.info(f"save i2t tta model at epoch {tta_epoch} to {os.path.join(registry.get_path('output_dir'), f'i2t_tta_model_model_epoch_{tta_epoch}.pth')}")

            logging.info("compute cosine similarity matrix, offline")
            sim_matrix_i2t, image_embeds, vit_feats, text_embeds, text_ids, text_atts = compute_embeds(tta_model.model, dataloader, task_cfg, tta_cfg)
            logging.info("compute i2t itm score offline, epoch %d :", tta_epoch)
            score_i2t= compute_i2t_itm_score(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts)
            results = report_metrics(scores_i2t=score_i2t, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics offline, at epoch {tta_epoch} :")
            logging.info(results)

    ## reset model to original state before t2i task
    tta_model.reset()
    ## t2i tta
    if tta_cfg.tta_task == "t2i":
        if tta_cfg.zero_shot_eval:
            logging.info("compute t2i itm score, zero-shot")
            score_t2i_zeroshot= compute_t2i_itm_score(tta_model.model, dataloader, tta_cfg, task_cfg, sim_matrix_t2i_zs, vit_feats_zs, text_ids_zs, text_atts_zs)
            result = report_metrics(scores_i2t=None, scores_t2i=score_t2i_zeroshot, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i metrics, zero-shot :")
            logging.info(result)
        ## multi epochs
        for tta_epoch in range(tta_cfg.offline_multi_epochs):
            logging.info(f"start t2i tta epoch {tta_epoch}")

            logging.info("adapt cosine similarity matrix, online")
            sim_matrix_i2t, sim_matrix_t2i, image_embeds, vit_feats, text_embeds, text_ids, text_atts = adapt_t2i_itc_sim(tta_model.model, dataloader, task_cfg, optimizer, tta_cfg)
            logging.info("compute t2i itm score online, epoch %d :", tta_epoch)
            score_t2i= compute_t2i_itm_score(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_t2i, vit_feats, text_ids, text_atts)
            results = report_metrics(scores_i2t=None, scores_t2i=score_t2i, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i metrics online, at epoch {tta_epoch} :")
            logging.info(results)
            # torch.save(tta_model.model.state_dict(), os.path.join(registry.get_path("output_dir"), f"t2i_tta_model_model_epoch_{tta_epoch}.pth"))
            # logging.info(f"save t2i tta model at epoch {tta_epoch} to {os.path.join(registry.get_path('output_dir'), f't2i_tta_model_model_epoch_{tta_epoch}.pth')}")

            logging.info("compute cosine similarity matrix, offline")
            sim_matrix_i2t, sim_matrix_t2i,  image_embeds, vit_feats, text_embeds, text_ids, text_atts = compute_embeds(tta_model.model, dataloader, task_cfg=task_cfg)
            logging.info("compute t2i itm score offline, epoch %d :", tta_epoch)
            score_t2i= compute_t2i_itm_score(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_t2i, vit_feats, text_ids, text_atts)
            results = report_metrics(scores_i2t=None, scores_t2i=score_t2i, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i metrics offline, at epoch {tta_epoch} :")
            logging.info(results)

    return score_i2t, score_t2i


def forward_and_itm_adapt_v2(tta_model, optimizer, dataloader, task_cfg, tta_cfg):
    score_i2t, score_t2i = None, None

    ## compute cosine similarity matrix
    logging.info("compute cosine similarity matrix")
    # sim_matrix_i2t, sim_matrix_t2i, image_embeds, vit_feats, text_embeds, text_ids, text_atts = compute_embeds(tta_model.model, dataloader, task_cfg, tta_cfg)
    # np.save("debugs/debug_sim_matrix_i2t.npy",sim_matrix_i2t.numpy())
    # np.save("debugs/debug_sim_matrix_t2i.npy",sim_matrix_t2i.numpy())
    # np.save("debugs/debug_image_embeds.npy",image_embeds.numpy())
    # np.save("debugs/debug_vit_feats.npy",vit_feats.numpy())
    # np.save("debugs/debug_text_embeds.npy",text_embeds.numpy())
    # np.save("debugs/debug_text_ids.npy",text_ids.numpy())
    # np.save("debugs/debug_text_atts.npy",text_atts.numpy())
    sim_matrix_i2t = torch.from_numpy(np.load("debugs/debug_sim_matrix_i2t.npy"))
    sim_matrix_t2i = torch.from_numpy(np.load("debugs/debug_sim_matrix_t2i.npy"))
    image_embeds = torch.from_numpy(np.load("debugs/debug_image_embeds.npy"))
    vit_feats = torch.from_numpy(np.load("debugs/debug_vit_feats.npy"))
    text_embeds = torch.from_numpy(np.load("debugs/debug_text_embeds.npy"))
    text_ids = torch.from_numpy(np.load("debugs/debug_text_ids.npy"))
    text_atts = torch.from_numpy(np.load("debugs/debug_text_atts.npy"))
    if is_main_process():
        result = report_metrics(scores_i2t=sim_matrix_i2t.numpy(), scores_t2i=sim_matrix_t2i.numpy(), txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report recall metrics, zero-shot :")
        logging.info(f"report recall metrics, zero-shot :")
        logging.info(result)

    ## compute augmentation cosine similarity matrix
    if tta_cfg.top1_match_coeffi == True and tta_cfg.top1_match_coeffi_src in ["KL_itm", "KL_itc"]:
        logging.info("compute augmentation cosine similarity matrix")
        # sim_matrix_i2t_aug, sim_matrix_t2i_aug, image_embeds_aug, vit_feats_aug, text_embeds_aug, text_ids_aug, text_atts_aug = compute_embeds(tta_model.model, dataloader, task_cfg, tta_cfg, is_aug=True)
        # np.save("debug_sim_matrix_i2t_aug.npy",sim_matrix_i2t_aug.numpy())
        # np.save("debug_sim_matrix_t2i_aug.npy",sim_matrix_t2i_aug.numpy())
        # np.save("debug_image_embeds_aug.npy",image_embeds_aug.numpy())
        # np.save("debug_vit_feats_aug.npy",vit_feats_aug.numpy())
        # np.save("debug_text_embeds_aug.npy",text_embeds_aug.numpy())
        # np.save("debug_text_ids_aug.npy",text_ids_aug.numpy())
        # np.save("debug_text_atts_aug.npy",text_atts_aug.numpy())
        sim_matrix_i2t_aug = torch.from_numpy(np.load("debugs/debug_sim_matrix_i2t_aug.npy"))
        sim_matrix_t2i_aug = torch.from_numpy(np.load("debugs/debug_sim_matrix_t2i_aug.npy"))
        image_embeds_aug = torch.from_numpy(np.load("debugs/debug_image_embeds_aug.npy"))
        vit_feats_aug = torch.from_numpy(np.load("debugs/debug_vit_feats_aug.npy"))
        text_embeds_aug = torch.from_numpy(np.load("debugs/debug_text_embeds_aug.npy"))
        text_ids_aug = torch.from_numpy(np.load("debugs/debug_text_ids_aug.npy"))
        text_atts_aug = torch.from_numpy(np.load("debugs/debug_text_atts_aug.npy"))
        if is_main_process():
            result = report_metrics(scores_i2t=sim_matrix_i2t_aug.numpy(), scores_t2i=sim_matrix_t2i_aug.numpy(), txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report recall metrics, aug zero-shot :")
            logging.info(f"report recall metrics, aug zero-shot :")
            logging.info(result)
        # {'txt_r1': 67.58, 'txt_r5': 90.34, 'txt_r10': 94.88, 'txt_r_mean': 84.26666666666667, 'txt_mAP': 57.78, 'img_r1': 58.972411035585765, 'img_r5': 82.86685325869652, 'img_r10': 89.31627349060376, 'img_r_mean': 77.05184592829535, 'img_mAP': 69.62, 'r_mean': 80.65925629748101}
        # {'txt_r1': 68.12, 'txt_r5': 90.54, 'txt_r10': 94.92, 'txt_r_mean': 84.52666666666669, 'txt_mAP': 58.01, 'img_r1': 59.21231507397041, 'img_r5': 82.83486605357857, 'img_r10': 89.23230707716914, 'img_r_mean': 77.09316273490604, 'img_mAP': 69.77, 'r_mean': 80.80991470078636}
        # {'txt_r1': 67.8, 'txt_r5': 90.24, 'txt_r10': 94.62, 'txt_r_mean': 84.22, 'txt_mAP': 57.91, 'img_r1': 58.82846861255498, 'img_r5': 82.89884046381448, 'img_r10': 89.42423030787685, 'img_r_mean': 77.0505131280821, 'img_mAP': 69.54, 'r_mean': 80.63525656404104}
        kl_coeffis_i2t, kl_coeffis_t2i = compute_kl_coeffis(sim_matrix_i2t, sim_matrix_i2t_aug)

    ## i2t tta
    if tta_cfg.tta_task == "i2t":
        if tta_cfg.zero_shot_eval:
            logging.info("compute i2t itm score, zero-shot")
            score_i2t_zeroshot, itm_score_i2t_zeroshot = compute_i2t_itm_score_v2(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts)

            if is_main_process():
                result = report_metrics(scores_i2t=score_i2t_zeroshot, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics, zero-shot :")
                logging.info(f"report i2t metrics, zero-shot :")
                logging.info(result)
        #     if is_main_process():
        #         result = report_metrics(scores_i2t=itm_score_i2t_zeroshot, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report itm i2t metrics, zero-shot :")
        #         logging.info(f"report itm i2t metrics, zero-shot :")
        #         logging.info(result)
        # ## multi epochs
        itm_score_list = []
        for tta_epoch in range(tta_cfg.offline_multi_epochs):
            logging.info(f"start i2t tta epoch {tta_epoch}")

            logging.info("adapt i2t itm score online, epoch %d :", tta_epoch)
            score_i2t, itm_score_i2t = adapt_i2t_itm_score_v2(tta_model.model, dataloader, task_cfg, optimizer, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts, tta_epoch)
            itm_score_list.append(itm_score_i2t)
            ## save epoch scores json
            if is_main_process():
                npy_path = os.path.join(registry.get_path("output_dir"), f"result/tta_epochs_score_distribution.npy")
                np.save(npy_path, np.concatenate(itm_score_list))
            plt_itm_score(np.concatenate(itm_score_list), task="i2t", mode="tta")

            if is_main_process():
                results = report_metrics(scores_i2t=score_i2t, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics online, epoch {tta_epoch} :")
                logging.info(f"report i2t metrics online, epoch {tta_epoch} :")
                logging.info(results)
            # if is_main_process():
            #     results = report_metrics(scores_i2t=itm_score_i2t, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t itm metrics online, epoch {tta_epoch} :")
            #     logging.info(f"report i2t itm metrics online, epoch {tta_epoch} :")
            #     logging.info(results)

            logging.info("compute i2t itm score offline, epoch %d :", tta_epoch)
            score_i2t, itm_score_i2t = compute_i2t_itm_score_v2(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_i2t, vit_feats, text_ids, text_atts, tta_epoch)

            if is_main_process():
                results = report_metrics(scores_i2t=score_i2t, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t metrics offline, epoch {tta_epoch} :")
                logging.info(f"report i2t metrics offline, epoch {tta_epoch} :")
                logging.info(results)
            # if is_main_process():
            #     results = report_metrics(scores_i2t=itm_score_i2t, scores_t2i=None, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report i2t itm metrics offline, epoch {tta_epoch} :")
            #     logging.info(f"report i2t itm metrics offline, epoch {tta_epoch} :")
            #     logging.info(results)

    # ## reset model to original state before t2i task
    tta_model.reset()
    ## t2i tta
    if tta_cfg.tta_task == "t2i":
        if tta_cfg.zero_shot_eval:
            logging.info("compute t2i itm score, zero-shot")
            score_t2i_zeroshot, itm_score_t2i_zeroshot = compute_t2i_itm_score_v2(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_t2i, vit_feats, text_ids, text_atts)

            if is_main_process():
                result = report_metrics(scores_i2t=None, scores_t2i=score_t2i_zeroshot, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i metrics, zero-shot :")
                logging.info(f"report t2i metrics, zero-shot :")
                logging.info(result)
            # if is_main_process():
            #     result = report_metrics(scores_i2t=None, scores_t2i=itm_score_t2i_zeroshot, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i itm metrics, zero-shot :")
            #     logging.info(f"report t2i itm metrics, zero-shot :")
            #     logging.info(result)
        ## multi epochs
        itm_score_list = []
        for tta_epoch in range(tta_cfg.offline_multi_epochs):
            logging.info(f"start t2i tta epoch {tta_epoch}")

            logging.info("adapt t2i itm score online, epoch %d :", tta_epoch)
            score_t2i, itm_score_t2i = adapt_t2i_itm_score_v2(tta_model.model, dataloader, task_cfg, optimizer, tta_cfg, sim_matrix_t2i, vit_feats, text_ids, text_atts, tta_epoch)
            itm_score_list.append(itm_score_t2i)
            if is_main_process():
                npy_path = os.path.join(registry.get_path("output_dir"), f"result/tta_epochs_score_distribution.npy")
                np.save(npy_path, np.concatenate(itm_score_list))
            plt_itm_score(np.concatenate(itm_score_list), task="t2i", mode="tta")

            if is_main_process():
                results = report_metrics(scores_i2t=None, scores_t2i=score_t2i, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i metrics online, epoch {tta_epoch} :")
                logging.info(f"report t2i metrics online, epoch {tta_epoch} :")
                logging.info(results)
            # if is_main_process():
            #     results = report_metrics(scores_i2t=None, scores_t2i=itm_score_t2i, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i itm metrics online, epoch {tta_epoch} :")
            #     logging.info(f"report t2i itm metrics online, epoch {tta_epoch} :")
            #     logging.info(results)

            logging.info("compute t2i itm score offline, epoch %d :", tta_epoch)
            score_t2i, itm_score_t2i = compute_t2i_itm_score_v2(tta_model.model, dataloader, task_cfg, tta_cfg, sim_matrix_t2i, vit_feats, text_ids, text_atts, tta_epoch)

            if is_main_process():
                results = report_metrics(scores_i2t=None, scores_t2i=score_t2i, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i metrics offline, epoch {tta_epoch} :")
                logging.info(f"report t2i metrics offline, epoch {tta_epoch} :")
                logging.info(results)
            # if is_main_process():
            #     results = report_metrics(scores_i2t=None, scores_t2i=itm_score_t2i, txt2img=dataloader.dataset.txt2img, img2txt=dataloader.dataset.img2txt, prefix_info=f"report t2i itm metrics offline, epoch {tta_epoch} :")
            #     logging.info(f"report t2i itm metrics offline, epoch {tta_epoch} :")
            #     logging.info(results)

    return score_i2t, score_t2i