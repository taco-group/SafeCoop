# README

To support heterogeneous multi-agent scenarios, where different ego vehicles utilize different Vision-Language Models (VLMs), you need to modify the following two components.

## Step 1
in the `simulation/leaderboard/team_code/agent_config/vlm_config_speed_curvature.yaml`

```yaml
simulation:
    ego_num: 2                     # number of communicating drivable ego vehicles
    skip_frames: 4                 # frame gap before a new driving control signal is generated

heter:
    avail_heter_planner_configs: ["vlmdrive/hypes_yaml/api_vlm_drive_speed_curvature_qwen2.5-72b-awq.yaml", "vlmdrive/hypes_yaml/api_vlm_drive_speed_curvature_qwen2.5-3b-awq.yaml", "vlmdrive/hypes_yaml/api_vlm_drive_speed_curvature_qwen2.5-7b-awq.yaml"]
    ego_planner_choice: [1, 2]
```

- `avail_heter_planner_configs`: Specifies all available VLM configurations.
- `ego_planner_choice`: Assigns a specific VLM to each ego vehicle. The index corresponds to the order in avail_heter_planner_configs.

Ensure that the `ego_num` parameter in the `simulation` section matches the number of ego vehicles you intend to simulate.

## Step 2
in the `scripts/eval_driving_vlm.sh`, make sure ego_num is correct.
```bash
export EGO_NUM=2
```

## Step 3 - Launch vLLM
Please refer to the doc `docs/vllm_ad.md`. We assign each model to a specific GPU. You need to launch all the models you intend to use.

- Ensure that each VLM instance runs on a unique port and that the port configurations align correctly.

```bash
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
    --download-dir /other/vlm_models \
    --host 0.0.0.0 \
    --port 8001 \ # port should be different and match with vlmdrive/hypes_yaml/api_vlm_drive_speed_curvature_qwen2.5-7b-awq.yaml
    --dtype float16 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 8192 \
    --trust-remote-code
```