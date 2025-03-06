Route_id=_all
Carla_port=2000
Method_tag="v1_single_no_obj_speed_curvature"
Repeat_id=0
Agent_config="speed_curvature"
Scenario_config="1"

CUDA_VISIBLE_DEVICES=0 bash scripts/eval_driving_vlm.sh ${Route_id} ${Carla_port} ${Method_tag} ${Repeat_id} ${Agent_config} ${Scenario_config}
