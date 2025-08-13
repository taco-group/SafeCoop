"""
Base class for Visual-Language Model based planners.
This abstract class defines the interface and common functionality for all VLM planners.
"""

from typing import List, Dict, Tuple, Any, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
import json
import numpy as np
from abc import ABC, abstractmethod
from vlmdrive.tools.status_tracker import StatusTracker
from vlmdrive.v2x_managers.v2x_managers import V2XManager

# Utils (support both relative and flat imports to avoid breakage)
try:
    from .vlm_planner_utils import (
        configure_vlm_helpers, ensure_list, get_ego_front_image, get_all_objects,
        get_related_pos_with_direction, build_collab_agent_description,
        get_collab_agent_images, build_target_description, STAGES
    )
except Exception:
    from vlm_planner_utils import (
        configure_vlm_helpers, ensure_list, get_ego_front_image, get_all_objects,
        get_related_pos_with_direction, build_collab_agent_description,
        get_collab_agent_images, build_target_description, STAGES
    )

# Set up logger
_logger = logging.getLogger(__name__)


class VLMPlannerBase(ABC, nn.Module):
    """
    Base class for Vision-Language Model based planners.
    Implements common functionality for scene understanding and trajectory planning.
    """

    def __init__(self, name: Any, api_model_name: Any, api_base_url: Any, api_key: Any, **kwargs):
        """
        Initialize the VLM planner.

        Args:
            name: Model name/path or a dict per stage
            api_model_name: API model name or a dict per stage
            api_base_url: API base url or a dict per stage
            api_key: API key (string or path) or a dict per stage
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"

        # Build helpers in a single place
        helpers = configure_vlm_helpers(
            name=name,
            api_model_name=api_model_name,
            api_base_url=api_base_url,
            api_key=api_key,
            image_placeholder=self.IMAGE_PLACEHOLDER,
        )
        # Keep same attributes as before
        self.vlm_helper_scene = helpers['scene']
        self.vlm_helper_object = helpers['object']
        self.vlm_helper_intention = helpers['intention']
        self.vlm_helper_target = helpers['target']
        self.vlm_helper_comb = helpers['comb']

        self.status_tracker = StatusTracker()
        self.v2x_manager = V2XManager()

    # -------------------- High-level perception-to-text --------------------

    def get_scene_description(self, obs_images, prompt_usage, prompt_template=None, idx=0):
        """
        Generate a description of the scene from the observation images.
        """
        template_version = prompt_usage["scene_prompt_template"]
        if not template_version:
            return "", ""
        prompt = prompt_template["scene_prompt_template"][template_version]
        result = self.vlm_inference(stage='scene', text=prompt, images=obs_images, idx=idx)
        print(f"Scene description Prompt: {prompt}, idx: {idx}")
        print(f"Scene description: {result}, idx: {idx}")
        return result, prompt

    def get_objects_description(self, obs_images, prompt_usage, prompt_template=None, idx=0):
        """
        Generate a description of objects in the scene.
        """
        template_version = prompt_usage["object_prompt_template"]
        if not template_version:
            return "", ""
        prompt = prompt_template["object_prompt_template"][template_version]
        result = self.vlm_inference(stage='object', text=prompt, images=obs_images, idx=idx)
        print(f"Object description Prompt: {prompt}, idx: {idx}")
        print(f"Object description: {result}, idx: {idx}")
        return result, prompt

    def get_intent_description(self, obs_images, prompt_usage, target_description=None, prompt_template=None, idx=0):
        """
        Generate or update the driving intention based on observations and target.
        """
        template_version = prompt_usage["intention_prompt_template"]
        if not template_version:
            return "", ""
        prompt = prompt_template["intention_prompt_template"][template_version].format(
            target_description=target_description
        )
        result = self.vlm_inference(stage='intention', text=prompt, images=obs_images, idx=idx)
        print(f"Intention description Prompt: {prompt}, idx: {idx}")
        print(f"Intention description: {result}, idx: {idx}")
        return result, prompt

    def get_target_description(self, target_waypoint, prompt_template=None, idx=0):
        """
        Generate a natural language description of the target waypoint relative to the vehicle.
        """
        prompt = build_target_description(target_waypoint, prompt_template, idx)
        return prompt

    # -------------------- Inference & helpers --------------------

    def vlm_inference(self, stage, text=None, images=None, sys_message=None, idx=0):
        """
        Run inference with the Vision-Language Model.
        """
        assert stage in STAGES, f"Invalid stage: {stage}"
        images = ensure_list(images)

        if stage == 'scene':
            return self.vlm_helper_scene.infer(images=images, text=text, sys_message=sys_message)
        if stage == 'object':
            return self.vlm_helper_object.infer(images=images, text=text, sys_message=sys_message)
        if stage == 'intention':
            return self.vlm_helper_intention.infer(images=images, text=text, sys_message=sys_message)
        if stage == 'target':
            return self.vlm_helper_target.infer(images=images, text=text, sys_message=sys_message)
        # 'comb'
        return self.vlm_helper_comb.infer(images=images, text=text, sys_message=sys_message)

    # ---- Delegated small helpers (preserve method names & signatures) ----

    def _get_ego_front_image(self, perception_memory_bank, agent_idx):
        """Delegate to utils; keep original signature."""
        return get_ego_front_image(perception_memory_bank, agent_idx)

    def _get_all_objects(self, perception_memory_bank):
        """Delegate to utils; keep original signature."""
        return get_all_objects(perception_memory_bank)

    def _get_related_pos_with_direction(self, ego_pos, ego_yaw, positions):
        """Delegate to utils; keep original signature."""
        return get_related_pos_with_direction(ego_pos, ego_yaw, positions)

    def _get_collab_agent_description(self, collab_agent_message, with_image=False) -> str:
        """Delegate to utils; keep original signature."""
        return build_collab_agent_description(collab_agent_message, self.IMAGE_PLACEHOLDER, with_image=with_image)

    def _get_collab_agent_image(self, perception_memory_bank, model_config, idx):
        """Delegate to utils; keep original signature."""
        return get_collab_agent_images(perception_memory_bank, model_config, idx)

    # -------------------- Planning core --------------------

    def _gen_individual_info(
        self,
        images,
        ego_history_prompt,
        model_config,
        scene_description,
        object_description,
        intent_description,
        target_description,
        collab_agent_description,
        idx=0,
    ):
        """
        Generates future speed-curvature pairs using the VLM.
        """
        # images, scene_description, object_description, intent_description, target_description, collab_agent_description, idx = \
        #     self._normalize_comb_inputs(image_or_list, ego_history_prompt, model_config, *args)

        comb_prompt = model_config["planning"]["prompt_template"]["comb_prompt"]["default"].format(
            scene_description=scene_description,
            object_description=object_description,
            intent_description=intent_description,
            target_description=target_description,
            ego_history_prompt=ego_history_prompt,
            collab_agent_description=collab_agent_description
        )
        sys_message = model_config["planning"]["prompt_template"]["sys_message"]

        for attempt in range(3):
            result = self.vlm_inference(stage='comb', text=comb_prompt, images=images, sys_message=sys_message, idx=idx)
            try:
                result = self._postprocess_result(result)
                break
            except Exception as e:
                if attempt < 2:
                    _logger.warning(f"Failed to parse JSON (attempt {attempt+1}/3): {str(e)}")
                else:
                    _logger.error("Failed to parse JSON after 3 attempts, returning empty waypoints.")
                    return None

        print(f"ego_history_prompt: {ego_history_prompt}")
        print(f"collab_agent_description: {collab_agent_description}")
        print(f"Predicted result for agent {idx}: {result}")
        return self._result_to_prediction_dict(result)

    # -------------------- Abstract hooks --------------------

    @abstractmethod
    def _result_to_prediction_dict(self, result):
        """Convert raw VLM output to planner prediction dict."""
        pass

    @abstractmethod
    def _get_ego_history(self, perception_memory_bank, agent_idx):
        """Return JSON string of ego history."""
        pass

    @abstractmethod
    def _postprocess_result(self, result):
        """Extract numeric outputs (e.g., [speed, curvature] pairs) from raw VLM response."""
        pass

    # -------------------- Batch flows --------------------

    def _forward_single_cot(self, perception_memory_bank, front_image_ego, model_config, idx):
        scene_description, _ = self.get_scene_description(
            front_image_ego,
            prompt_usage=model_config["planning"]["prompt_usage"],
            prompt_template=model_config["planning"]["prompt_template"],
            idx=idx
        )
        object_description, _ = self.get_objects_description(
            front_image_ego,
            prompt_usage=model_config["planning"]["prompt_usage"],
            prompt_template=model_config["planning"]["prompt_template"],
            idx=idx
        )
        return scene_description, object_description

    def forward_single_intent(self, perception_memory_bank, model_config, idx):
        """
        Processes a single intent.
        """
        if "intent" not in model_config['collab']['sharing_modalities']:
            return None

        front_image_ego = self._get_ego_front_image(perception_memory_bank, idx)
        scene_description, object_description = self._forward_single_cot(perception_memory_bank, front_image_ego, model_config, idx)
        target_waypoint = perception_memory_bank[-1]["target"][idx]
        target_description = self.get_target_description(
            target_waypoint=target_waypoint,
            prompt_template=model_config["planning"]["prompt_template"],
            idx=idx
        )
        intent_description, _ = self.get_intent_description(
            front_image_ego,
            target_description=target_description,
            prompt_usage=model_config["planning"]["prompt_usage"],
            prompt_template=model_config["planning"]["prompt_template"],
            idx=idx
        )
        return {
            "idx": idx,
            "position": perception_memory_bank[-1]["localization"][idx],
            "ego_yaw": perception_memory_bank[-1]["ego_yaw"][idx],
            "scene_description": scene_description,
            "object_description": object_description,
            "target_description": target_description,
            "intent_description": intent_description,
        }

    def forward_single_collab(
        self,
        perception_memory_bank,
        model_config,
        idx,
        collab_agent_message=[]
    ):
        """
        Processes collaborative agent information.
        """
        TRANSMIT_IMG = False
        
        front_image_ego = self._get_ego_front_image(perception_memory_bank, idx)
        # scene_description, object_description = self._forward_single_cot(perception_memory_bank, front_image_ego, model_config, idx)
        front_images_dict = self._get_collab_agent_image(perception_memory_bank, model_config, idx)

        scene_description, object_description, target_description, intent_description = "", "", "", ""
        collab_agent_message_collected = []
        for message in collab_agent_message:
            if message is None:
                continue
            if front_images_dict and message["idx"] in front_images_dict and message["idx"] != idx:
                if TRANSMIT_IMG:
                    message.update({"front_image": front_images_dict[message["idx"]]})
                message['position'] = self._get_related_pos_with_direction(
                    ego_pos=perception_memory_bank[-1]["localization"][idx],
                    ego_yaw=perception_memory_bank[-1]["ego_yaw"][idx],
                    positions=message['position']
                )
                collab_agent_message_collected.append(message)
            if message["idx"] == idx:
                scene_description = message.get("scene_description", "")
                object_description = message.get("object_description", "")
                target_description = message.get("target_description", "")
                intent_description = message.get("intent_description", "")

        collab_agent_message_collected = sorted(collab_agent_message_collected, key=lambda x: x['idx'])
        
        
        ################ Attack and Defense Simulation ################
        attacked_message = self.v2x_manager.simulate_attack(collab_agent_message_collected, 
                                                            ego_idx=idx)
        defensed_message, malicious_ids = self.v2x_manager.simulate_defense(attacked_message, 
                                                                            ego_idx=idx)
        
        
        ################ Postprocess Messages ################
        collab_agent_description = self._get_collab_agent_description(collab_agent_message_collected, with_image=TRANSMIT_IMG)
        all_image_list = [front_image_ego]
        if TRANSMIT_IMG:
            all_image_list.extend([it.get("front_image", None) for it in collab_agent_message_collected])

        ego_history_prompt = self._get_ego_history(perception_memory_bank, idx)
        # target_waypoint = perception_memory_bank[-1]["target"][idx]
        # target_description = self.get_target_description(
        #     target_waypoint=target_waypoint,
        #     prompt_template=model_config["planning"]["prompt_template"],
        #     idx=idx
        # )

        pred_result = self._gen_individual_info(
            all_image_list,
            ego_history_prompt,
            model_config,
            scene_description,
            object_description,
            intent_description,
            target_description,
            collab_agent_description,
            idx=idx
        )
        return pred_result