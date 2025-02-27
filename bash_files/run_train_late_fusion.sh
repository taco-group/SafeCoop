# python opencood/tools/train.py \
#     -y opencood/hypes_yaml/v2xverse/late_fusion_multiclass_config.yaml
    # [--model_dir ${CHECKPOINT_FOLDER}]


# CUDA_VISIBLE_DEVICES=0,1 \
#     python -m torch.distributed.launch  \
#     --nproc_per_node=2 \
#     --use_env opencood/tools/train_ddp.py \
#     -y opencood/hypes_yaml/v2xverse/late_fusion_multiclass_config.yaml

python opencood/tools/train.py \
    -y opencood/hypes_yaml/v2xverse/lss_single_multiclass_config.yaml
    # [--model_dir ${CHECKPOINT_FOLDER}]
