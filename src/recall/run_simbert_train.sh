python train_simbert.py \
    --model_name simbert \
    --train_batch_size 48 \
    --epoch 5 \
    --learning_rate 1e-5 \
    --use_ecom \
    --use_video > simbert.lr5e.ecom.video.log 2>&1 &
    # --fgm > 
