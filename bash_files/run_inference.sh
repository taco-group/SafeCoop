python opencood/tools/inference.py \
    --model_dir opencood/logs/v2xverse_codriving_ms_atten_multiclass_2025_01_27_12_17_06
    # -y opencood/hypes_yaml/v2xverse/codriving_multiclass_config.yaml
    # [--model_dir ${CHECKPOINT_FOLDER}]


# CUDA_VISIBLE_DEVICES=0,1 \
#     python -m torch.distributed.launch  \
#     --nproc_per_node=2 \
#     --use_env opencood/tools/train_ddp.py \
#     -y opencood/hypes_yaml/v2xverse/codriving_multiclass_config.yaml