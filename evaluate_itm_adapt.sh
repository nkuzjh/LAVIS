#!/bin/bash

# 设置基础参数
num_gpus=3
log_dir="./logs"
mkdir -p $log_dir

# 定义主端口列表（每个任务使用不同的端口）
ports=(29500 29501 29502 29503 29504 29505)

# 定义训练脚本路径（根据实际情况修改）
scripts=(
    "evaluate_tta.py"
    "evaluate_tta.py"
    "evaluate_tta.py"
    "evaluate_tta.py"
    "evaluate_tta.py"
    "evaluate_tta.py"
)

# 定义日志文件名
log_files=(
    "ret_coco_eval_itm_adapt_i2t_exp9.1.3.1.out"
    "ret_coco_eval_itm_adapt_i2t_exp9.1.3.2.out"
    "ret_coco_eval_itm_adapt_i2t_exp9.1.3.3.out"
    "ret_coco_eval_itm_adapt_i2t_exp9.1.3.4.out"
    "ret_coco_eval_itm_adapt_i2t_exp9.1.3.5.out"
    "ret_coco_eval_itm_adapt_i2t_exp9.1.3.6.out"
)

# 定义参数文件名
args_files=(
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.1.yaml"
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.2.yaml"
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.3.yaml"
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.4.yaml"
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.5.yaml"
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.6.yaml"
)

# 按顺序执行每个训练任务
for i in "${!scripts[@]}"
do
    script=${scripts[$i]}
    log_file=$log_dir/${log_files[$i]}
    arg_file=${args_files[$i]}
    port=${ports[$i]}

    echo "Starting training: $script on port $port"
    # nohup torchrun --nproc_per_node=$num_gpus --master_port=$port $script > $log_file 2>&1 &
    nohup python -m torch.distributed.run --nproc_per_node=$num_gpus --master_port=$port $script --is_tta True --cfg-path $arg_file > $log_file 2>&1 &
    
    # 等待当前任务完成
    wait
    echo "Training $arg_file completed."
done

echo "All training tasks completed."

# CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.1.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.1.out 2>&1 & ;

# CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.2.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.2.out 2>&1 & ;

# CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.3.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.3.out 2>&1 & ;

# CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.4.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.4.out 2>&1 & ;

# CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.5.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.5.out 2>&1 & ;

# CUDA_VISIBLE_DEVICES=0,1,2 nohup python -m torch.distributed.run --nproc_per_node=3 evaluate_tta.py --is_tta True --cfg-path lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp9.1.3.6.yaml > ret_coco_eval_itm_adapt_i2t_exp9.1.3.6.out 2>&1 & ;