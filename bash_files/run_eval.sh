Route_id=1
Carla_port=2000
Method_tag="test"
Repeat_id=0
Agent_config="coopenemma"
Scenario_config="1"

CUDA_VISIBLE_DEVICES=1 bash scripts/eval_driving_vlm.sh ${Route_id} ${Carla_port} ${Method_tag} ${Repeat_id} ${Agent_config} ${Scenario_config}