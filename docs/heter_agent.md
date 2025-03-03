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