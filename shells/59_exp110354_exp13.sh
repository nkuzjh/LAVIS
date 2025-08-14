#!/bin/bash

# 设置基础参数
# num_gpus=1
log_dir="./logs"
mkdir -p $log_dir

# 定义主端口列表（每个任务使用不同的端口）
# ports=(
#     29500
#     29501
#     29502
#     29503
#     29504
#     # 29505
# )

# 定义训练脚本路径（根据实际情况修改）
scripts=(
    "tta_i2t.py"
    "tta_i2t.py"
    "tta_i2t.py"
    "tta_i2t.py"
    "tta_i2t.py"
    "tta_i2t.py"

)

# 定义日志文件名
log_files=(
    # "ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.out"
    # "ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.1.out"
    # "ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.2.out"
    # "ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.3.out"

    "ret_coco_eval_itm_adapt_i2t_exp13"
    "ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.4.out"
    "ret_coco_eval_itm_adapt_i2t_exp13.1"
    "ret_coco_eval_itm_adapt_i2t_exp13.2"
    "ret_coco_eval_itm_adapt_i2t_exp13.3"
    "ret_coco_eval_itm_adapt_i2t_exp13.4"

)

# 定义参数文件名
args_files=(
    # "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.yaml"
    # "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.1.yaml"
    # "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.2.yaml"
    # "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.3.yaml"

    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp13.yaml"
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp11.0.3.5.4.yaml"
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp13.1.yaml"
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp13.2.yaml"
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp13.3.yaml"
    "lavis/projects/blip2/eval/ret_coco_eval_itm_adapt_i2t_exp13.4.yaml"

)

# 按顺序执行每个训练任务
for i in "${!scripts[@]}"
do
    script=${scripts[$i]}
    log_file=$log_dir/${log_files[$i]}
    arg_file=${args_files[$i]}
    # port=${ports[$i]}

    echo " "
    echo "Starting training: $script $arg_file on port $port"
    start_time=$(date +%s)
    echo "Start Time: $(date +"%Y-%m-%d %T")"

    # nohup torchrun --nproc_per_node=$num_gpus --master_port=$port $script > $log_file 2>&1 &
    # nohup python -m torch.distributed.run --nproc_per_node=$num_gpus --master_port=$port $script --is_tta True --cfg-path $arg_file > $log_file 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup python $script --is_tta True --cfg-path $arg_file > $log_file 2>&1 &

    # 等待当前任务完成
    wait

    # 记录结束时间
    end_time=$(date +%s)
    duration=$(( end_time - start_time ))
    # 格式化时间（分钟和秒）
    minutes=$(( duration / 60 ))
    seconds=$(( duration % 60 ))
    # 结束时间和运行时长
    echo "End Time: $(date +"%Y-%m-%d %T")"
    echo "Duration: ${minutes}m${seconds}s"
    echo "Training $script $arg_file completed."
done

echo "All training tasks completed."