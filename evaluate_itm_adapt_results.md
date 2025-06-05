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
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out 2>&1 &    58 running pid=1019782  
    - **results :**
        - i2t best:
            report i2t metrics offline, at epoch 2 / 9:
            {"txt_r1": 85.02, "txt_r5": 96.64, "txt_r10": 98.24, "txt_r_mean": 93.3, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
        - t2i best:
            report t2i metrics online, at epoch 2 / 4:
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.37704918032787, "img_r5": 87.38104758096762, "img_r10": 92.54698120751699, "img_r_mean": 82.43502598960417, "r_mean": -999, "agg_metrics": -999}

1. exp 0.1
    - loss = Qformer tent; top1_match_coeffi=False
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 1
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp01.out 2>&1 &    58 
        **验证i2t top1_match_coeffi为何不生效**
    - **results :**
        - i2t best:
            report i2t metrics offline, at epoch 2 / 9:
            {"txt_r1": 85.02, "txt_r5": 96.64, "txt_r10": 98.24, "txt_r_mean": 93.3, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}



1. exp 1
    - loss = Qformer tent / exp(1-互相top1-topk的概率均值); top1_match_coeffi=True
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp1.out 2>&1 &    58 stop due to bug, in which the result is same to exp0 i2t
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp1.out 2>&1 &    59 finish, also same to the exp0 i2t. But the bug has already been excluded, and it can be see that in the running t2i exp0&1 the results are different.
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i_exp1.out 2>&1 &    59 running pid=2036707
    - **results :**
        - i2t best: 
            report i2t metrics offline, at epoch 2 / 9 :
            {"txt_r1": 85.02, "txt_r5": 96.64, "txt_r10": 98.24, "txt_r_mean": 93.3, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
        - t2i best: 
            report t2i metrics offline, at epoch 1 / 1:
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.44502199120352, "img_r5": 87.52099160335865, "img_r10": 92.5749700119952, "img_r_mean": 82.5136612021858, "r_mean": -999, "agg_metrics": -999}


1. exp 2
    - sample selection of inter top1
    - loss = Qformer tent / exp(1-互相top1-topk的概率均值); top1_match_coeffi=True
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_ss.yaml > ret_coco_eval_itm_adapt_i2t_exp2.out 2>&1 &    58 running pid=2122497
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i_ss.yaml > ret_coco_eval_itm_adapt_t2i_exp2.out 2>&1 &    59 running pid=2871505
        **remote 59的CUDA_VISIBLE_DEVICES=2是不是没挂载上？**
    - **results :**
        - i2t best: 
        - t2i best: 