python generate_rank_encode_data.py \
    --recall_model roformer_sim_ft \
    --recall_base_model roformer \
    --rank_model simbert \
    --pooling_type cls \
    --use_ecom \
    --use_video \
    --recall_model_path /root/autodl-tmp/alibaba/data/models/roformer_sim_ft_sample_10w_0_lr1e-05_drop0.0_pt_cls_ecom_True_video_True_fgm_False_lrdecay_False_augfalse_hard9_/epoch0 \
    --outdir /root/autodl-tmp/alibaba/data/rank_data_0296/ > gen_data.log 2>&1 &
    