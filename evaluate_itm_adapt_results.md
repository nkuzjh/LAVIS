# evaluate_itm_adapt.py Results


## command
- debug args: --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml --is_tta True
- CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t.out 2>&1 &
- CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out 2>&1 &


## baseline score
1. official inference code:
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --cfg-path lavis/projects/blip2/eval/ret_coco_eval.yaml >ret_coco_eval.out 2>&1 & 
        58 finish
        _report_metrics: 
            {"txt_r1": 85.42, "txt_r5": 97.02, "txt_r10": 98.48, "txt_r_mean": 93.64, "img_r1": 68.25269892043183, "img_r5": 87.72890843662535, "img_r10": 92.62694922031187, "img_r_mean": 82.869518859123, "r_mean": 88.2547594295615, "agg_metrics": 93.64}
2. my implemention:
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_sigmoid.yaml > ret_coco_eval_itm_adapt_i2t_exp3.out 2>&1 & 
        report zero-shot metrics with origin function : 
            {"txt_r1": 84.82, "txt_r5": 96.62, "txt_r10": 98.22, "txt_r_mean": 93.21999999999998, "img_r1": 67.37305077968813, "img_r5": 87.20511795281887, "img_r10": 92.41903238704518, "img_r_mean": 82.33240037318406, "r_mean": 87.77620018659202, "agg_metrics": 93.21999999999998}
        report i2t zero-shot metrics : 
            {"txt_r1": 84.82, "txt_r5": 96.56, "txt_r10": 98.3, "txt_r_mean": 93.22666666666667, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
        report t2i zero-shot metrics : 
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.36505397840864, "img_r5": 87.21311475409836, "img_r10": 92.42702918832467, "img_r_mean": 82.33506597361055, "r_mean": -999, "agg_metrics": -999}
    - --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_my_implementation_debug.yaml
        58 finish debugging
        **update model.eval() in eval funtion to fix diff of bsl**
        report i2t zero-shot metrics : 
            {"txt_r1": 85.42, "txt_r5": 97.02, "txt_r10": 98.48, "txt_r_mean": 93.64, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
        report zero-shot metrics with origin function : 
            {"txt_r1": 85.42, "txt_r5": 97.02, "txt_r10": 98.48, "txt_r_mean": 93.64, "img_r1": 68.25269892043183, "img_r5": 87.72890843662535, "img_r10": 92.62694922031187, "img_r_mean": 82.869518859123, "r_mean": 88.2547594295615, "agg_metrics": 93.64}
        report t2i zero-shot metrics : 
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 68.25269892043183, "img_r5": 87.72890843662535, "img_r10": 92.62694922031187, "img_r_mean": 82.869518859123, "r_mean": -999, "agg_metrics": -999}
3. diff of baseline and my imple, due to lack of model.eval() in compute_sim_matrix function;


## experiments
### exp 0
1. parameters:
    - loss = Qformer tent; top1_match_coeffi=False
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
2. i2t best results:
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t.out 2>&1 &    
        58 finish
        report i2t metrics offline, at epoch 2 / 9:
            {"txt_r1": 85.02, "txt_r5": 96.64, "txt_r10": 98.24, "txt_r_mean": 93.3, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t.out1 2>&1 &   
        58 finish
        report i2t metrics offline, at epoch 7 / 9:
            {"txt_r1": 85.02, "txt_r5": 96.34, "txt_r10": 98.1, "txt_r_mean": 93.15333333333335, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t.out2 2>&1 &  
        **update model.eval() in eval funtion to fix diff of bsl**
        58 finish;
        report i2t metrics offline, at epoch 3 / 9:
            {"txt_r1": **85.5**, "txt_r5": 96.9, "txt_r10": 98.42, "txt_r_mean": 93.60666666666667, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
3. t2i best results:
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out 2>&1 &    
        58 stop at epoch=4 without reason 
        report t2i metrics online, at epoch 2 / 4:
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.37704918032787, "img_r5": 87.38104758096762, "img_r10": 92.54698120751699, "img_r_mean": 82.43502598960417, "r_mean": -999, "agg_metrics": -999}
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out1 2>&1 &    
        58 killed by nobody? running again below
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i.out1 2>&1 &    
        58 finish
        report t2i metrics offline, at epoch 1 / 9:
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.4610155937625, "img_r5": 87.44102359056377, "img_r10": 92.57097161135546, "img_r_mean": 82.49100359856057, "r_mean": -999, "agg_metrics": -999}


### exp 0.1
1. parameters:
    - loss = Qformer tent / exp(1-互相top1-topk的概率均值); top1_match_coeffi=True
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 1
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
2. i2t best results:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp01.out 2>&1 &    
        58 finish
        **验证i2t top1_match_coeffi为何不生效**
        **当gradacc很大时，由于一个iter的loss变小，导致每次梯度更新结果相同**
        report i2t metrics online, at epoch 0 / 9 :
            {"txt_r1": 84.44, "txt_r5": 96.28, "txt_r10": 98.3, "txt_r_mean": 93.00666666666666, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}


### exp 1
1. parameters:
    - loss = Qformer tent / exp(1-互相top1-topk的概率均值); top1_match_coeffi=True
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp1.out 2>&1 &    
        58 stop at epoch=4, in which the result is same to exp0 i2t; and the training log is same to i2t_exp1 in 59 below;
        report i2t metrics offline, at epoch 2 / 4:
            {"txt_r1": 85.02, "txt_r5": 96.64, "txt_r10": 98.24, "txt_r_mean": 93.3, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp1.out 2>&1 &    
        59 finish, also same to the exp0 i2t. But the bug has already been excluded, and it can be see that in the running t2i exp0&1 the results are different.
        report i2t metrics offline, at epoch 2 / 9 :
            {"txt_r1": 85.02, "txt_r5": 96.64, "txt_r10": 98.24, "txt_r_mean": 93.3, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t.yaml > ret_coco_eval_itm_adapt_i2t_exp1.out1 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        58 finish
        report i2t metrics offline, at epoch 3 / 9:
            {"txt_r1": **85.5**, "txt_r5": 96.9, "txt_r10": 98.42, "txt_r_mean": 93.60666666666667, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
3. results t2i best:
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i_exp1.out 2>&1 &    
        59 stop at epoch=3 without reason
        report t2i metrics offline, at epoch 2 / 3:
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.48500599760096, "img_r5": 87.28108756497402, "img_r10": 92.34306277489004, "img_r_mean": 82.36971877915501, "r_mean": -999, "agg_metrics": -999}
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i.yaml > ret_coco_eval_itm_adapt_t2i_exp1.out1 2>&1 &    
        **59 CUDA_VISIBLE_DEVICES=2 实际是 CUDA_VISIBLE_DEVICES=1的3090;**
        **59 CUDA_VISIBLE_DEVICES=1 实际是 CUDA_VISIBLE_DEVICES=2 的4090**
        59 finish;
        **update model.eval() in eval funtion to fix diff of bsl**
        report t2i metrics offline, at epoch 0 / 9:
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": **68.26069572171131**, "img_r5": 87.76489404238305, "img_r10": 92.63094762095162, "img_r_mean": 82.885512461682, "r_mean": -999, "agg_metrics": -999}


### exp 1.1.1/visual_log_debug
1. paramenters:
    - 增加日志： 训练loss
    - 增加日志： indicator exp(1 - inter_topk_match_proba) 与 正负样本 的关系
2. i2t debugging:
    - --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_visual_log_debug.yaml
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_visual_log_debug.yaml > ret_coco_eval_itm_adapt_i2t_visual_log_debug.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        59 finish;
        report i2t metrics offline, at epoch 3 / 9:
            {"txt_r1": 85.5, "txt_r5": 96.9, "txt_r10": 98.42, "txt_r_mean": 93.60666666666667, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
        **entropy和loss曲线没有下降，只是抖动**
     - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-1-1.yaml > ret_coco_eval_itm_adapt_i2t_exp1.1.1.out1 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        修改了记录log和曲线图的方法
        58 finish
        report i2t metrics offline, at epoch 3 / 9:
            {"txt_r1": 85.5, "txt_r5": 96.9, "txt_r10": 98.42, "txt_r_mean": 93.60666666666667, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
     - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_visual_log_debug.yaml > ret_coco_eval_itm_adapt_i2t_visual_log_debug.out1 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        修改了记录log和曲线图的方法
        仅记录zero-shot的log
        58 debugging device=0

### exp 1.1.2
1. paramenters:
    - 增加日志： 训练loss
    - 增加日志： indicator exp(1 - inter_topk_match_proba) 与 正负样本 的关系
    - exp改为中的inter_topk_match_proba改为由sigmoid计算产生
2. i2t debugging:
    - --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-1-2.yaml
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-1-2.yaml > ret_coco_eval_itm_adapt_i2t_1.1.2.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        58 finish
        report i2t metrics offline, at epoch 3 / 9:
            {"txt_r1": 85.5, "txt_r5": 96.9, "txt_r10": 98.42, "txt_r_mean": 93.60666666666667, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
        **entropy和loss曲线没有下降，只是抖动**
        **且改变inter_topk_match_proba为sigmoid也对结果无影响**
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-1-2.yaml > ret_coco_eval_itm_adapt_i2t_1.1.2.out1 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        修改了记录log和曲线图的方法
        58 finish
        report i2t metrics offline, at epoch 3 / 9:
            {"txt_r1": 85.5, "txt_r5": 96.9, "txt_r10": 98.42, "txt_r_mean": 93.60666666666667, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}

### exp 1.0.1
1. parameters:
    - lr=1e-4 wd=0.0
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-0-1.yaml > ret_coco_eval_itm_adapt_i2t_exp1.0.1.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        58 finish;
        效果不好, epoch0=84.+
3. results t2i best:
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i_exp1-0-1.yaml > ret_coco_eval_itm_adapt_t2i_exp1.0.1.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        59 finish
        report t2i metrics offline, at epoch 0 / 9:
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.32906837265094, "img_r5": 87.29708116753298, "img_r10": 92.47101159536186, "img_r_mean": 82.36572037851526, "r_mean": -999, "agg_metrics": -999}
        效果不好, epoch0=67.+
### exp 1.0.2
1. parameters:
    - lr=3.5e-4 wd=0.0
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-0-2.yaml > ret_coco_eval_itm_adapt_i2t_exp1.0.2.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        58 finish;
        效果不好, epoch0=82.+
### exp 1.0.3
1. parameters:
    - lr=1e-3 wd=0.0
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-0-3.yaml > ret_coco_eval_itm_adapt_i2t_exp1.0.3.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        58 finish;
        效果不好, epoch0=82.+
### exp 1.0.4
1. parameters:
    - lr=5e-5 wd=0.0
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-0-4.yaml > ret_coco_eval_itm_adapt_i2t_exp1.0.4.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        58 finish;
        效果不好, epoch0=85.3
### exp 1.0.5
1. parameters:
    - lr=1e-5 wd=0.0
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-0-5.yaml > ret_coco_eval_itm_adapt_i2t_exp1.0.5.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        58 finish;
        report i2t metrics offline, at epoch 1 :
            {"txt_r1": 85.44, "txt_r5": 96.9, "txt_r10": 98.42, "txt_r_mean": 93.58666666666666, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
### exp 1.0.6
1. parameters:
    - lr=1e-6 wd=0.0
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-0-6.yaml > ret_coco_eval_itm_adapt_i2t_exp1.0.6.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        58 finish;
        report i2t metrics offline, at epoch 1 :
            {"txt_r1": 85.46, "txt_r5": 97.02, "txt_r10": 98.48, "txt_r_mean": 93.65333333333332, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
### exp 1.0.7
1. parameters:
    - lr=5e-6 wd=0.0
    - grad_accum_num: = 128
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-0-7.yaml > ret_coco_eval_itm_adapt_i2t_exp1.0.7.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        59 running pid=1091726
        report i2t metrics offline, at epoch 0 / 7:
            {"txt_r1": 85.46, "txt_r5": 97.0, "txt_r10": 98.48, "txt_r_mean": 93.64666666666666, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
### exp 1.0.8
1. parameters:
    - lr=5e-6 wd=1e-4
    - grad_accum_num: = 64
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-0-8.yaml > ret_coco_eval_itm_adapt_i2t_exp1.0.8.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**
        59 finish;
        report i2t metrics offline, at epoch 3 / 9:
            {"txt_r1": 85.5, "txt_r5": 96.9, "txt_r10": 98.42, "txt_r_mean": 93.60666666666667, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
### exp 1.0.9
1. parameters:
    - lr=5e-6 wd=0.0
    - grad_accum_num: = 32
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp1-0-9.yaml > ret_coco_eval_itm_adapt_i2t_exp1.0.9.out 2>&1 & 
        **update model.eval() in eval funtion to fix diff of bsl**


### exp 2
1. parameters:
    - sample selection of inter top1
    - loss = Qformer tent / exp(1-互相top1-topk的概率均值); top1_match_coeffi=True
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_ss.yaml > ret_coco_eval_itm_adapt_i2t_exp2.out 2>&1 &    
        58 stop at epoch 5, oom
        report i2t metrics offline, at epoch 2 / 5:
            {"txt_r1": 85.02, "txt_r5": 96.62, "txt_r10": 98.28, "txt_r_mean": 93.30666666666666, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_ss.yaml > ret_coco_eval_itm_adapt_i2t_exp2.out 2>&1 &    
        58 stop at epoch 5, oom
        report i2t metrics offline, at epoch 2 / 5:
            {"txt_r1": 85.02, "txt_r5": 96.62, "txt_r10": 98.28, "txt_r_mean": 93.30666666666666, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_ss.yaml > ret_coco_eval_itm_adapt_i2t_exp2.out 2>&1 &  
        58 stop at epoch 5, oom
        report i2t metrics offline, at epoch 2 / 5:
            {"txt_r1": 85.02, "txt_r5": 96.62, "txt_r10": 98.28, "txt_r_mean": 93.30666666666666, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
3. results t2i best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i_ss.yaml > ret_coco_eval_itm_adapt_t2i_exp2.out 2>&1 &    
        58 finish
        report t2i metrics offline, at epoch 4 / 9:
            {"txt_r1": -999, "txt_r5": -999, "txt_r10": -999, "txt_r_mean": -999, "img_r1": 67.4970011995202, "img_r5": 87.47301079568173, "img_r10": 92.49500199920033, "img_r_mean": 82.48833799813409, "r_mean": -999, "agg_metrics": -999}


### exp 3
1. parameters:
    - loss = Qformer sigmoid entropy; top1_match_coeffi=False
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 1
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_sigmoid.yaml > ret_coco_eval_itm_adapt_i2t_exp3.out 2>&1 &    
        59 finish;
        epoch=0, 0.81+; 
        不收敛, 继续迭代exp 3.1,修改loss和其他参数;
3. results t2i best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i_sigmoid.yaml > ret_coco_eval_itm_adapt_t2i_exp3.out 2>&1 &    
        59 stop due to bug;


### exp 3.1
1. parameters:
    - loss = Qformer sigmoid entropy; top1_match_coeffi=False
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_sigmoid.yaml > ret_coco_eval_itm_adapt_i2t_exp3.1.out 2>&1 &   
        59 finish
        **update model.eval() in eval funtion to fix diff of bsl**
        report i2t metrics offline, at epoch 0 / 9 :
            {"txt_r1": 85.42, "txt_r5": 97.0, "txt_r10": 98.48, "txt_r_mean": 93.63333333333334, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
3. results t2i best:   
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i_sigmoid.yaml > ret_coco_eval_itm_adapt_t2i_exp3.out 2>&1 &    
        59 debug


### exp 3.1.1
1. parameters:
    - lr=1e-4 wd=0.0
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_sigmoid.yaml > ret_coco_eval_itm_adapt_i2t_exp3.1.1.out 2>&1 &   
        59 stop 
        **update model.eval() in eval funtion to fix diff of bsl**
        不收敛
### exp 3.1.2
1. parameters:
    - lr=3.5e-4 wd=0.0
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_sigmoid.yaml > ret_coco_eval_itm_adapt_i2t_exp3.1.2.out 2>&1 &   
        58 stop
        **update model.eval() in eval funtion to fix diff of bsl**
        不收敛
### exp 3.1.3
1. parameters:
    - lr=8e-4 wd=0.0
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_sigmoid.yaml > ret_coco_eval_itm_adapt_i2t_exp3.1.3.out 2>&1 &   
        58 finish;
        **update model.eval() in eval funtion to fix diff of bsl**
        不收敛


### exp 3.2
1. parameters:
    - loss = Qformer sigmoid entropy / tta_cfg.temper; top1_match_coeffi=False
    - tta_cfg.temper = 0.01
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_sigmoid_exp3-2.yaml > ret_coco_eval_itm_adapt_i2t_exp3.2.out 2>&1 &   
        59 kill for JUHAO pid=4140412; 
        **update model.eval() in eval funtion to fix diff of bsl**
        59 finish;
        report i2t metrics offline, at epoch 0 / 9:
            {"txt_r1": 85.42, "txt_r5": 97.0, "txt_r10": 98.48, "txt_r_mean": 93.63333333333334, "img_r1": -999, "img_r5": -999, "img_r10": -999, "img_r_mean": -999, "r_mean": -999, "agg_metrics": -999}
3. results t2i best:   
    - CUDA_VISIBLE_DEVICES=1 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_t2i_sigmoid.yaml > ret_coco_eval_itm_adapt_t2i_exp3.out 2>&1 &    
        59 debug


### exp 3.3
1. parameters:
    - loss = Qformer sigmoid entropy .mean() ; top1_match_coeffi=False; temper=1
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 1
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - log_iters=50
    - **update model.eval() in eval funtion to fix diff of bsl**
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp3-3.yaml > ret_coco_eval_itm_adapt_i2t_exp3.3.out 2>&1 &   
        58 finish; 不收敛
### exp 3.3.1
1. parameters:
    - loss = Qformer sigmoid entropy .mean() ; top1_match_coeffi=False; temper=1
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - log_iters=50
    - **update model.eval() in eval funtion to fix diff of bsl**
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=2 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp3-3-1.yaml > ret_coco_eval_itm_adapt_i2t_exp3.3.1out 2>&1 &   
        58 running pid=4071532


### exp 4
1. parameters:
    - loss = Qformer itm_logits softmax entropy; top1_match_coeffi=False; temper=1
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 1
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - log_iters=50
    - **update model.eval() in eval funtion to fix diff of bsl**
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp4.yaml > ret_coco_eval_itm_adapt_i2t_exp4.out 2>&1 &   
        58 running pid=4084980
    
### exp 4.0.1
1. parameters:
    - loss = Qformer itm_logits softmax entropy; top1_match_coeffi=False; temper=1
    - lr=5e-6 wd=0.0
    - 梯度累积, accumulate batch size = 64
    - offline, online=False
    - rerank_score = itm_score +　cosine similarity without dividing temperature(cos_sim in rerank tta without dividing temperature)
    - multi_epochs=10
    - log_iters=50
    - **update model.eval() in eval funtion to fix diff of bsl**
2. results i2t best:
    - CUDA_VISIBLE_DEVICES=0 nohup python evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp4-0-1.yaml > ret_coco_eval_itm_adapt_i2t_exp4.0.1.out 2>&1 &   
        59