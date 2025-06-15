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

        for _ in range(self.steps):
            if tta_cfg.name == 'itm_adapt' or tta_cfg.name == 'visenc_itm_adapt' or tta_cfg.name == 'all_itm_adapt':
                outputs = forward_and_itm_adapt(self, self.optimizer, data_loader, task_cfg, tta_cfg)
            elif tta_cfg.name == 'itm_adapt_ss':
                outputs = forward_and_itm_adapt_ss(self, self.optimizer, data_loader, task_cfg, tta_cfg)
            elif tta_cfg.name == 'itm_adapt_sigmoid':
                outputs = forward_and_itm_adapt_sigmoid(self, self.optimizer, data_loader, task_cfg, tta_cfg)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

@torch.jit.script
def zhh_softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return (-(F.softmax(x) * F.log_softmax(x))).sum()

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
    agg_metrics = -999

    if scores_i2t is not None:
        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        tr_mean = (tr1 + tr5 + tr10) / 3

    if scores_t2i is not None:
        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        ir_mean = (ir1 + ir5 + ir10) / 3

    if scores_i2t is not None and scores_t2i is not None:
        r_mean = (tr_mean + ir_mean) / 2
        agg_metrics = (tr1 + tr5 + tr10) / 3

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
        "agg_metrics": agg_metrics,
    }
    with open(
        os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
    ) as f:
        f.write(prefix_info + "\n")
        f.write(json.dumps(eval_result) + "\n")
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