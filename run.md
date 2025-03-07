# README

## how to run
```bash
# launch carla
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=2000 -prefer-nvidia

# run homo
bash bash_files/run_eval_api_speed_curvature_homo.sh

# run heter
bash bash_files/run_eval_api_speed_curvature_heter.sh
```