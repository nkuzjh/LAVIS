# evaluate_itm_adapt.py Results

## command
- debug args: --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml --is_tta True
- CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t.out 2>&1 &
- CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out 2>&1 &

## experiments
1. exp 0
    - loss = Qformer tent; top1_match_coeffi=False
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t.out 2>&1 &    58 finish
        - **results :**
        - i2t best:
            report i2t metrics offline, at epoch 2 / 9:
            {"txt_r1": 85.02, "txt_r5": 96.64, "txt_r10": 98.24, "txt_r_mean": 93.3, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out 2>&1 &    58 stop at epoch=4 without reason 
        - **results :**
        - t2i best:
            report t2i metrics online, at epoch 2 / 4:
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.37704918032787, "img_r5": 87.38104758096762, "img_r10": 92.54698120751699, "img_r_mean": 82.43502598960417, "r_mean": -999, "agg_metrics": -999}

    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t.out1 2>&1 &    58 finish
        - **results :**
         - i2t best:
            report i2t metrics online, at epoch 4 :
            {"txt_r1": 84.94, "txt_r5": 96.44, "txt_r10": 98.14, "txt_r_mean": 93.17333333333333, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}

    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out1 2>&1 &    58 running pid=2787432; killed by nobody? running again below
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out1 2>&1 &    58 running again pid=3352611



1. exp 0.1
    - loss = Qformer tent / exp(1-互相top1-topk的概率均值); top1_match_coeffi=True
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 1
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp01.out 2>&1 &    58 finish
        **验证i2t top1_match_coeffi为何不生效**
        **当gradacc很大时，由于一个iter的loss变小，导致每次梯度更新结果相同**
        - **results :**
        - i2t best:
            report i2t metrics online, at epoch 0 / 9 :
            {"txt_r1": 84.44, "txt_r5": 96.28, "txt_r10": 98.3, "txt_r_mean": 93.00666666666666, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}



1. exp 1
    - loss = Qformer tent / exp(1-互相top1-topk的概率均值); top1_match_coeffi=True
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp1.out 2>&1 &    58 stop at epoch=4, in which the result is same to exp0 i2t; and the training log is same to i2t_exp1 in 59 below;
        - **results :**
        -i2t best:
            report i2t metrics offline, at epoch 2 :
            {"txt_r1": 85.02, "txt_r5": 96.64, "txt_r10": 98.24, "txt_r_mean": 93.3, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}


    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp1.out 2>&1 &    59 finish, also same to the exp0 i2t. But the bug has already been excluded, and it can be see that in the running t2i exp0&1 the results are different.
        - **results :**
        - i2t best: 
            report i2t metrics offline, at epoch 2 / 9 :
            {"txt_r1": 85.02, "txt_r5": 96.64, "txt_r10": 98.24, "txt_r_mean": 93.3, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}

    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i_exp1.out 2>&1 &    59 stop at epoch=3 without reason
        - **results :**
        - t2i best: 
            report t2i metrics offline, at epoch 2 / 3:
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.48500599760096, "img_r5": 87.28108756497402, "img_r10": 92.34306277489004, "img_r_mean": 82.36971877915501, "r_mean": -999, "agg_metrics": -999}

    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i_exp1.out1 2>&1 &    59 running pid = 3690453;
        **remote 59的CUDA_VISIBLE_DEVICES=2是不是没挂载上？**
        **59 CUDA_VISIBLE_DEVICES=2 实际是 CUDA_VISIBLE_DEVICES=1的3090;**
        - **results :**
        - t2i best: 

1. exp 2
    - sample selection of inter top1
    - loss = Qformer tent / exp(1-互相top1-topk的概率均值); top1_match_coeffi=True
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_ss.yaml > ret_coco_eval_itm_adapt_i2t_exp2.out 2>&1 &    58 running pid=3140541
        - **results :**
        - i2t best: 

    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i_ss.yaml > ret_coco_eval_itm_adapt_t2i_exp2.out 2>&1 &    58 running pid=3363937
        - **results :**
        - t2i best: 


1. exp 3
    - loss = Qformer sigmoid entropy; top1_match_coeffi=False
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 1
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_sigmoid.yaml > ret_coco_eval_itm_adapt_i2t_exp3.out 2>&1 &    59 running pid=4042538
        **59 CUDA_VISIBLE_DEVICES=1 实际是 CUDA_VISIBLE_DEVICES=2 的4090**
        - **results :**
        - i2t best: 

    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i_sigmoid.yaml > ret_coco_eval_itm_adapt_t2i_exp3.out 2>&1 &    59 running pid=4044806
        - **results :**
        - t2i best: 