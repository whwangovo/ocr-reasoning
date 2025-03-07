# Two GPUs are left for vLLM inference acceleration.
# pip install math_verify # reward function
# pip install git+https://github.com/huggingface/trl.git
# GPU memory: 8 * 60GiB

export WANDB_PROJECT="swift_ocr_r1"

TIMESTAMP=$(date '+%y%m%d_%H%M%S')
PROJECT_NAME="Qwen2VL@base@2B@open-r1-8k@${TIMESTAMP}"
RUN_NAME=$PROJECT_NAME


MAX_PIXELS=602112 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model checkpoints/Qwen/VL/Qwen2-VL-2B \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset lmms-lab/multimodal-open-r1-8k-verified \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-7 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 6 \
    --temperature 1.2 \
    --top_p 0.9 \
    --top_k 50 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true \
    --report_to tensorboard wandb \
    --run_name ${RUN_NAME}
    # --async_generate true \
    # --num_iterations 1 \
    # --num_infer_workers 2 \
