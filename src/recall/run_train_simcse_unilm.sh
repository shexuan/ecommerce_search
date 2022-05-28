python train_simcse_unilm.py \
    --model_name roformer \
    --train_batch_size 48 \
    --epoch 5 \
    --learning_rate 1e-5 \
    --use_ecom \
    --use_video \
    --final_activation tanh \
    --suffix simcse_unilm > simcse_unilm.1e5.log 2>&1 &
    