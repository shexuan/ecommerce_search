python trainer.py \
    --n_neg 2 \
    --rank_idx 9 \
    --batch_size 48 \
    --num_epochs 5 \
    --learning_rate 1e-5 \
    --warmup_proportion 0 \
    --max_seq_length 128 \
    --save_steps 10000 \
    --eval_steps 10 \
    --label_smooth 0.1 \
    --is_training \
    --model_name robert \
    --data_dir /root/autodl-tmp/alibaba/data/robert_rank_data_0296 \
    --output_dir /root/autodl-tmp/alibaba/data/rank_models/ > robert.rank9.neg2.novideo.log 2>&1 &
    
    # --suffix new \