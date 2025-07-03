#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.1.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.1.out 2>&1 & ;

CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.2.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.2.out 2>&1 & ;

CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.3.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.3.out 2>&1 & ;

CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.4.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.4.out 2>&1 & ;

CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.5.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.5.out 2>&1 & ;

CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.6.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.6.out 2>&1 & ;

