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
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t.out 2>&1 &    58
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out 2>&1 &    58
    - **results :**
        - i2t  
        - i2t  

1. exp 1
    - loss = Qformer tent / exp(1-互相top1-topk的概率均值); top1_match_coeffi=True
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp1.out 2>&1 &    58 stop due to bug
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp1.out 2>&1 &    59
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i_exp1.out 2>&1 &    59 running pid=2036707
    - **results :**
        - i2t best: 
            report i2t metrics offline, at epoch 2 :
            {"txt_r1": 85.02, "txt_r5": 96.64, "txt_r10": 98.24, "txt_r_mean": 93.3, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
        - i2t best: 