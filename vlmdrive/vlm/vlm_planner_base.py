"""
Base class for Visual-Language Model based planners.
This abstract class defines the interface and common functionality for all VLM planners.
"""

from typing import List, Dict, Tuple, Any
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
import base64
import os.path
import re
import json
from datetime import datetime
import math
import numpy as np
from PIL import Image
import httpx
from pathlib import Path
from abc import ABC, abstractmethod
from vlmdrive.vlm_api_helper import VLMAPIHelper

# Set up logger
_logger = logging.getLogger(__name__)

class VLMPlannerBase(ABC, nn.Module):
    """
    Base class for Vision-Language Model based planners.
    Implements common functionality for scene understanding and trajectory planning.
    """
    
    def __init__(self, name: str, api_model_name: str, api_base_url: str, api_key: str, **kwargs):
        """
        Initialize the VLM planner.
        
        Args:
            name: Model name/path
            api_model_name: Name of the API model to use
            api_base_url: Base URL for the API
            api_key: API key or path to a file containing the API key
            **kwargs: Additional keyword arguments
        """
        super().__init__()

        if "api" in name:
            
            self.IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"
            
            self.model_path = name
            # Support both file path and direct API key
            try:
                self.api_key = Path(api_key).read_text().strip()
                _logger.info(f"API key loaded from {api_key}")
            except:
                self.api_key = api_key
                _logger.info("API key loaded from direct input")
                
            self.vlm_helper = VLMAPIHelper(api_key=self.api_key, api_base_url=api_base_url, api_model_name=api_model_name, image_placeholder=self.IMAGE_PLACEHOLDER)
        else:
            raise ValueError(f"Unsupported model path: {name}")
        
        
    def get_scene_description(self, obs_images, prompt_usage, prompt_template=None, idx=0):
        """
        Generate a description of the scene from the observation images.
        
        Args:
            obs_images: Images of the scene
            prompt_template: Template for prompts
            
        Returns:
            tuple: (scene description, prompt used)
        """
        template_version = prompt_usage["scene_prompt_template"]
        if not template_version:
            return "", ""
        prompt = prompt_template["scene_prompt_template"][template_version]
        result = self.vlm_inference(text=prompt, images=obs_images, idx=idx)
        return result, prompt

    def get_objects_description(self, obs_images, prompt_usage, prompt_template=None, idx=0):
        """
        Generate a description of objects in the scene.
        
        Args:
            obs_images: Images of the scene
            prompt_template: Template for prompts
            
        Returns:
            tuple: (object description, prompt used)
        """
        template_version = prompt_usage["object_prompt_template"]
        if not template_version:
            return "", ""
        prompt = prompt_template["object_prompt_template"][template_version]
        result = self.vlm_inference(text=prompt, images=obs_images, idx=idx)
        return result, prompt

    def get_intent_description(self, obs_images, prompt_usage, target_description=None, prompt_template=None, idx=0):
        """
        Generate or update the driving intention based on observations and target.
        
        Args:
            obs_images: Images of the scene
            target_description: Description of the target location
            prompt_template: Template for prompts
            
        Returns:
            tuple: (intention description, prompt used)
        """
        template_version = prompt_usage["intention_prompt_template"]
        if not template_version:
            return "", ""
        prompt = prompt_template["intention_prompt_template"][template_version].format(target_description=target_description)
        result = self.vlm_inference(text=prompt, images=obs_images, idx=idx)
        return result, prompt
    
    def get_target_description(self, target_waypoint, prompt_template=None):
        """
        Generate a natural language description of the target waypoint relative to the vehicle.
        
        Args:
            target_waypoint: Coordinates of the target [x, y]
            prompt_template: Template for the target description
            
        Returns:
            str: Natural language description of target location
        """
        x = target_waypoint[0]
        y = -target_waypoint[1]
        
        prompt_template = prompt_template["target_prompt_template"]["default"]
        
        x_direction = "right" if x > 0 else "left"
        y_direction = "front" if y > 0 else "back"
            
        x_distance = abs(round(x, 5))
        y_distance = abs(round(y, 5))
        
        prompt = prompt_template.format(
            x_distance=x_distance, 
            x_direction=x_direction, 
            y_distance=y_distance, 
            y_direction=y_direction
        )
        
        return prompt
    

    def _gen_individual_info(self, 
                             image, 
                             ego_history_prompt, 
                             model_config, 
                             scene_description,
                             object_description,
                             intent_description,
                             target_description,
                             collab_agent_description,
                             idx=0):
        """
        Generates future speed-curvature pairs using the VLM.
        """
        
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
            result = self.vlm_inference(text=comb_prompt, images=image, sys_message=sys_message, idx=idx)
            try:
                result = self._postprocess_result(result)
                break
            except Exception as e:
                if attempt < 2:
                    _logger.warning(f"Failed to parse JSON (attempt {attempt+1}/3): {str(e)}")
                    # import traceback; traceback.print_exc()
                    # import pdb; pdb.set_trace()
                else:
                    _logger.error(f"Failed to parse JSON after 3 attempts, returning empty waypoints.")
                    return None
        return self._result_to_prediction_dict(result)

    def vlm_inference(self, text=None, images=None, sys_message=None, idx=0):
        """
        Run inference with the Vision-Language Model.
        """
            
        if not isinstance(images, list):
            images = [images]

        if "api" in self.model_path:
            return self.vlm_helper.infer(images=images, text=text, sys_message=sys_message)
        else:
            raise ValueError(f"Unsupported model path: {self.model_path}")
       
        
    def _get_ego_front_image(self, perception_memory_bank, agent_idx):
        """
        Get the front camera image of the ego vehicle.
        
        Args:
            perception_memory_bank: Historical perception data
            agent_idx: Index of the ego agent
            
        Returns:
            numpy.ndarray: RGB image from front camera
        """
        return perception_memory_bank[-1]['rgb_front'][agent_idx]  # (H, W, 3)
    
    
    def _get_all_objects(self, perception_memory_bank):
        """
        Get all detected objects including bounding boxes from CAV.
        
        Args:
            perception_memory_bank: Historical perception data
            
        Returns:
            list: Detected objects with bounding boxes
        """
        return perception_memory_bank[-1]["object_list"]


    @abstractmethod
    def _result_to_prediction_dict(self, result):
        """
        Convert the raw output from the VLM into a dictionary of predictions.
        
        Args:
            result: Raw output from the VLM
            
        Returns:
            dict: Dictionary of predictions
        """
        # This is an abstract method that should be implemented by subclasses
        pass
        

    @abstractmethod
    def _get_ego_history(self, perception_memory_bank, agent_idx):
        """
        Get the ego vehicle's history from the perception memory bank.
        
        Args:
            perception_memory_bank: Historical perception data
            agent_idx: Index of the ego agent
            
        Returns:
            str: JSON string with ego vehicle's history
        """
        # This is an abstract method that should be implemented by subclasses
        pass


    @abstractmethod
    def _postprocess_result(self, result):
        """
        Process the raw output from the VLM to extract speed and curvature values.
        
        Args:
            result: Raw output from the VLM
            init_x, init_y, init_yaw: Initial position and orientation
            dt: Time step between predictions
            
        Returns:
            numpy.ndarray: Array of [speed, curvature] pairs
        """
        # This is an abstract method that should be implemented by subclasses
        pass


    def forward(self, perception_memory_bank, model_config) -> torch.Tensor:
        """
        Perform the forward pass of the planner. For each ego vehicle in the batch, this method:
         1. Retrieves the latest captured front image.
         2. Prepares the ego history prompt.
         3. Generates waypoint predictions using the vision-language model.

        Args:
            perception_memory_bank (list[dict]): List of perception records.
            model_config (dict): Model configuration parameters.

        Returns:
            list: A list of predicted results for each ego vehicle.
        """
        # Determine the number of ego vehicles based on the front image shape
        N = perception_memory_bank[-1]["rgb_front"].shape[0]
        predicted_result_list = []

        for i in range(N):
            # (1) Retrieve the latest front image for the ego vehicle
            front_image = self._get_ego_front_image(perception_memory_bank, i)  # (H, W, 3)
            
            # (2) Prepare the ego history prompt
            ego_history_prompt = self._get_ego_history(perception_memory_bank, i)
            
            # (3) Generate waypoint predictions via the vision-language model
            pred_result = self._gen_individual_info(
                front_image,
                ego_history_prompt,
                model_config,
                perception_memory_bank,
                i
            )
            predicted_result_list.append(pred_result)
            
        return predicted_result_list
    
    
    def _forward_single_cot(self, perception_memory_bank, model_config, idx):
        front_image = self._get_ego_front_image(perception_memory_bank, idx)
        scene_description, _ = self.get_scene_description(front_image, 
                                                          prompt_usage=model_config["planning"]["prompt_usage"],
                                                          prompt_template=model_config["planning"]["prompt_template"],
                                                          idx=idx)
        object_description, _ = self.get_objects_description(front_image,
                                                             prompt_usage=model_config["planning"]["prompt_usage"],
                                                             prompt_template=model_config["planning"]["prompt_template"],
                                                             idx=idx)
        return scene_description, object_description
    
    
    def forward_single_intent(self, perception_memory_bank, model_config, idx):
        """
        Processes a single intent.
        
        Retrieves the ego front image, target description, and generates an intent description.
        """
        if "intent" not in model_config['collab']['sharing_modalities']:
            return None

        front_image = self._get_ego_front_image(perception_memory_bank, idx)
        target_waypoint = perception_memory_bank[-1]["target"][idx]
        target_description = self.get_target_description(target_waypoint=target_waypoint, 
                                                         prompt_template=model_config["planning"]["prompt_template"])
        intent_description, _ = self.get_intent_description(front_image, 
                                                            target_description=target_description,
                                                            prompt_usage=model_config["planning"]["prompt_usage"],
                                                            prompt_template=model_config["planning"]["prompt_template"],
                                                            idx=idx)
        return {
            "idx": idx,
            "position": perception_memory_bank[-1]["detmap_pose"][idx][:2],
            "intent_description": intent_description,
        }


    def _get_related_pos_with_direction(self, ego_pos, ego_yaw, positions):
        """
        Args:
            ego_pos (torch.Tensor or np.ndarray): The global position of the ego vehicle, shape (2,).
            ego_yaw (float): The yaw angle of the ego vehicle (in radians).
            positions (torch.Tensor or np.ndarray): The global positions of other vehicles, shape (N, 2).

        Returns:
            np.ndarray: The relative positions of other vehicles with respect to the ego vehicle's forward direction, shape (N, 2).
        """
        # Ensure input is a NumPy array (convert from PyTorch tensor if needed)
        if isinstance(ego_pos, torch.Tensor):
            ego_pos = ego_pos.cpu().numpy()
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()

        # Compute position difference (relative to ego vehicle)
        relative_global_pos = positions - ego_pos

        # Construct the rotation matrix
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        # rotation_matrix = np.array([[cos_yaw, -sin_yaw],
        #                             [sin_yaw,  cos_yaw]])
        rotation_matrix = np.array([
            [ sin_yaw,  cos_yaw],  # This row effectively becomes the local x
            [-cos_yaw,  sin_yaw]   # This row effectively becomes the local y
        ])

        # Apply rotation to obtain relative position
        relative_local_pos = relative_global_pos @ rotation_matrix.T

        return relative_local_pos


    def _get_collab_agent_description(self, collab_agent_intent) -> str:
        """
        Generates a description string for collaborative agents.
        
        For each collaborative agent, append its description text followed by an IMAGE_PLACEHOLDER.
        Note: This description only includes N-1 agents (excluding the ego agent whose image is handled separately).
        """
        description = ""
        for intent in collab_agent_intent:
            position = intent.get('position', '')
            
            # Convert position to a human-readable format
            if isinstance(position, torch.Tensor):
                position = position.detach().cpu().numpy()
            if isinstance(position, np.ndarray):
                position = position.tolist()
            
            # Round position values to 5 decimal places
            if isinstance(position, list):
                position = [round(coord, 5) for coord in position]
            
            description += (f"Agent {intent['idx']}, located at: {position}, "
                            f"intent description: {intent.get('intent_description', '')}, "
                            f"image: {self.IMAGE_PLACEHOLDER}\n")
        
        return description

    
    def _get_collab_agent_image(self, perception_memory_bank, model_config, idx):
        N = perception_memory_bank[-1]["rgb_front"].shape[0]
        if "image" in model_config['collab']['sharing_modalities']:
            front_images = perception_memory_bank[-1]["rgb_front"]
            front_images_dict = {}
            # Collect front images for all agents except the ego agent.
            for agent_idx in range(N):
                if agent_idx != idx:
                    front_images_dict[agent_idx] = front_images[agent_idx]
        else:
            front_images_dict = None
        return front_images_dict

    def forward_single_collab(self, 
                              perception_memory_bank, 
                              model_config, 
                              idx, 
                              collab_agent_intent=[]):
        """
        Processes collaborative agent information.
        
        - Retrieves the ego agent's image and the front images of the other agents.
        - Updates each collaborative agent's intent with its respective front image and adjusted position.
        - Generates a collab_agent_description string that interleaves each agent's text description with an IMAGE_PLACEHOLDER.
        
        Note: The ego agent's image is not included in the collab_agent_description.
        """

        front_image_ego = self._get_ego_front_image(perception_memory_bank, idx)
        scene_description, object_description = self._forward_single_cot(perception_memory_bank, model_config, idx)
        
        front_images_dict = self._get_collab_agent_image(perception_memory_bank, model_config, idx)
        
        intent_description = ""
        collab_agent_intent_new = [] # collab_agent_intent without ego agent
        for intent in collab_agent_intent:
            if intent is None:
                continue
            # Update each collaborative agent's intent with its respective front image and adjusted position.
            if front_images_dict and intent["idx"] in front_images_dict and intent["idx"] != idx:
                intent.update({
                    "front_image": front_images_dict[intent["idx"]]
                })
                intent['position'] = self._get_related_pos_with_direction(
                    ego_pos=perception_memory_bank[-1]["detmap_pose"][idx][:2],
                    ego_yaw=perception_memory_bank[-1]["ego_yaw"][idx],
                    positions=intent['position']
                )
                collab_agent_intent_new.append(intent)
            if intent["idx"] == idx:
                # Update the ego agent's intent description.
                intent_description = intent.get("intent_description", "")
            
        # sort collab_agent_intent by idx
        collab_agent_intent_new = sorted(collab_agent_intent_new, key=lambda x: x['idx'])
        # Generate the collaborative agents' description with image placeholders inserted.
        collab_agent_description = self._get_collab_agent_description(collab_agent_intent_new)
        all_image_list = [front_image_ego]
        all_image_list.extend([intent.get("front_image", None) for intent in collab_agent_intent_new])
        ego_history_prompt = self._get_ego_history(perception_memory_bank, idx)
        target_waypoint = perception_memory_bank[-1]["target"][idx]
        target_description = self.get_target_description(target_waypoint=target_waypoint, 
                                                         prompt_template=model_config["planning"]["prompt_template"])
        pred_result = self._gen_individual_info(all_image_list, 
                                                ego_history_prompt, 
                                                model_config, 
                                                scene_description,
                                                object_description,
                                                intent_description,
                                                target_description,
                                                collab_agent_description,
                                                idx)

        return pred_result
    
    
    
    