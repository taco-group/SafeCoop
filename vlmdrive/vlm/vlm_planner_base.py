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
            self.model_path = name
            self.api_model_name = api_model_name
            self.api_base_url = api_base_url
            
            # Support both file path and direct API key
            try:
                self.api_key = Path(api_key).read_text().strip()
                _logger.info(f"API key loaded from {api_key}")
            except:
                self.api_key = api_key
                _logger.info("API key loaded from direct input")
        else:
            raise ValueError(f"Unsupported model path: {name}")
        
        self.image_buffer = None
        
    def get_scene_description(self, obs_images, prompt_usage, prompt_template=None):
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
        result = self.vlm_inference(text=prompt, images=obs_images)
        return result, prompt

    def get_objects_description(self, obs_images, prompt_usage, prompt_template=None):
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
        result = self.vlm_inference(text=prompt, images=obs_images)
        return result, prompt

    def get_intent_description(self, obs_images, prompt_usage, target_description=None, prompt_template=None):
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
        result = self.vlm_inference(text=prompt, images=obs_images)
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

    def _gen_individual_info(self, image, ego_history_prompt, model_config, perception_memory_bank, agent_idx):
        """
        Generates future speed-curvature pairs using the VLM.
        """
        scene_description, _ = self.get_scene_description(image, 
                                                          prompt_usage=model_config["planning"]["prompt_usage"],
                                                          prompt_template=model_config["planning"]["prompt_template"])
        object_description, _ = self.get_objects_description(image,
                                                             prompt_usage=model_config["planning"]["prompt_usage"],
                                                             prompt_template=model_config["planning"]["prompt_template"])
        
        target_waypoint = perception_memory_bank[-1]["target"][agent_idx]
        target_description = self.get_target_description(target_waypoint=target_waypoint, 
                                                         prompt_template=model_config["planning"]["prompt_template"])
        
        intent_description, _ = self.get_intent_description(image, 
                                                            target_description=target_description,
                                                            prompt_usage=model_config["planning"]["prompt_usage"],
                                                            prompt_template=model_config["planning"]["prompt_template"])
        
        comb_prompt = model_config["planning"]["prompt_template"]["comb_prompt"]["default"].format(
            scene_description=scene_description, 
            object_description=object_description,
            intent_description=intent_description,
            target_description=target_description,
            ego_history_prompt=ego_history_prompt
        )
        
        sys_message = model_config["planning"]["prompt_template"]["sys_message"]

        for attempt in range(3):
            result = self.vlm_inference(text=comb_prompt, images=image, sys_message=sys_message)
            try:
                result = self._postprocess_result(result)
                break
            except Exception as e:
                if attempt < 2:
                    _logger.warning(f"Failed to parse JSON (attempt {attempt+1}/3): {str(e)}")
                else:
                    _logger.error(f"Failed to parse JSON after 3 attempts, returning empty waypoints.")
                    return None
        return self._result_to_prediction_dict(result)

    def vlm_inference(self, text=None, images=None, sys_message=None):
        """
        Run inference with the Vision-Language Model.
        """
        
        if self.image_buffer:
            images = self.image_buffer
        else:
            if isinstance(images, np.ndarray):
                images = Image.fromarray(images)
                save_dir = Path(os.environ['RESULT_ROOT']) / "image_buffer"
                save_dir.mkdir(exist_ok=True)
                image_dir = save_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-buffer.png"
                images.save(image_dir)
                images = str(image_dir)
                self.image_buffer = images
            else:
                raise ValueError("No valid image provided for inference.")

        if "api" in self.model_path:
            from openai import OpenAI

            os.environ['HTTPX_PROXIES'] = ''
            os.environ['no_proxy'] = '*'

            try:
                http_client = httpx.Client(transport=httpx.HTTPTransport(retries=3))
            except:
                http_client = httpx.Client(transport=httpx.HTTPTransport(retries=3))

            client = OpenAI(base_url=self.api_base_url, api_key=self.api_key, http_client=http_client)

            if isinstance(images, str):
                images = [images]

            content = [{"type": "text", "text": text}]
            for img_path in images:
                if os.path.exists(img_path):
                    with open(img_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        })
            
            messages = [{"role": "user", "content": content}]
            if sys_message:
                messages.insert(0, {"role": "system", "content": sys_message})

            params = {"model": self.api_model_name, "messages": messages, "max_tokens": 2048}
            result = client.chat.completions.create(**params)
            return result.choices[0].message.content
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
            
        self.image_buffer = None
            
        return predicted_result_list