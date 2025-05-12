python open_instruct/finetune.py \
    --exp_name olmo2_7b_sft \
    --model_name_or_path allenai/OLMo-2-1124-7B \
    --model_revision main \
    --tokenizer_name allenai/OLMo-2-1124-7B \
    --tokenizer_revision main \
    --use_slow_tokenizer False \
    --add_bos \
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --reduce_loss sum \
    --with_tracking \
    --logging_steps 1 \
    --seed 8
#--use_flash_attn \
#--report_to wandb \