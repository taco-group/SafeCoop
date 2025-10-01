Route_id=1
Carla_port=2020
Method_tag="safecoop_4agents_gpt_4_1_lights_on"
Repeat_id=0
Agent_config="safecoop/safecoop_4agents_gpt_4_1_openai"
Scenario_config="1_lights_on"

CUDA_VISIBLE_DEVICES=0 bash scripts/eval_driving_vlm.sh ${Route_id} ${Carla_port} ${Method_tag} ${Repeat_id} ${Agent_config} ${Scenario_config}