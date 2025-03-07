Route_id=_partial
Carla_port=20006
Method_tag="waypoints_CoT_concise_image_intent_2agent"
Repeat_id=0
Agent_config="waypoints_CoT_concise_image_intent_2agent"
Scenario_config="1"

CUDA_VISIBLE_DEVICES=0 bash scripts/eval_driving_vlm.sh ${Route_id} ${Carla_port} ${Method_tag} ${Repeat_id} ${Agent_config} ${Scenario_config}
