 python simsce_supervised.py \
     --model_name roformer_sim_ft \
     --base_model roformer \
     --pooling_type cls \
     --mode train \
     --sample_num 0 \
     --dropout_rate 0 \
     --train_batch_size 64 \
     --epoch 6 \
     --learning_rate 1e-5 > roformer.lr1e.drop0.fla.ecom.video.fp16.log 2>&1 &
#     --use_ecom \
#     --use_video > roformer.lr1e.drop0.fla.ecom.video.fp16.log 2>&1 &
#     # --suffix  
#     # --fgm 
#     # --aug random 
