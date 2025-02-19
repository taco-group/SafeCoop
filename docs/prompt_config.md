# Prompt Config

In this paper, we have three types of prompts: 
1. scene description
2. object description
3. shared and ego historical observation description

And we synthesize all prompts into one and feed to LLM to generate future waypoints.

## Scene/Object Description
The scene and object prompt templates can be configured in the file `simulation/leaderboard/team_code/agent_config/vlm_config_coopenemma_api.yaml`

## Ego Synthesized Description
Currently, this prompt includes:

1. a historical sequence of ego positions with timestamps.
2. detected objects from connected autonomous vehicles (CAVs).
3. the historical target destination of waypoints.

To modify the prompt, update the `_generate_vlm_prompt function` in `vlmdrive/vlm/base_vlm_planner.py`. You can use any information in `perception_memory_bank`, currently it contains (1) rgb images (front/left/right/rear) of all vehicles, (2) detected objects of each vehicle, (3) pose of each detection position and (4) the target waypoints of ego at each timestamp.

## Composing the Final Prompt
In the same configuration file as the Scene/Object Description, you can combine all the individual prompts mentioned above. The final prompt is then fed to the LLM. To do this, you only need to modify the `comb_prompt` template.