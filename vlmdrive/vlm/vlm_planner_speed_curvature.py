from typing import Iterable
from typing import Tuple, Dict

from vlmdrive import VLMDRIVE_REGISTRY

import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
_logger = logging.getLogger(__name__)


import base64
import os.path
import os
import re
import argparse
from datetime import datetime
from math import atan2
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from pyquaternion import Quaternion
from scipy.integrate import cumulative_trapezoid

import json
from vlmdrive.utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import httpx
from pathlib import Path

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN


@VLMDRIVE_REGISTRY.register
class VLMWaypointPlannerSpeedCurvature(nn.Module):
    """
    WaypointPlanner language prompt
    """
    
    
    def __init__(self, name, api_model_name, api_base_url, api_key, **kwargs):
        super().__init__()

        if "api" in name:
            self.model_path = name
            self.api_model_name = api_model_name
            self.api_base_url = api_base_url
            # Support both file path and API key
            try:
                self.api_key = Path(api_key).read_text().strip()
                print(f"API key loaded from {api_key}")
            except:
                self.api_key = api_key
                print(f"API key loaded from input")

        self.model_path = name
        
        # TODO: More VLMs and VLLM quantization

        # FIXME: temp dir for model weights
        cache_dir = "/4tb_ssd1/yuheng/model_cache"

        if "llama" in self.model_path:
            from transformers import MllamaForConditionalGeneration
            
            model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=cache_dir
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.tokenizer=None
            
        elif "qwen" in self.model_path:
            from qwen_vl_utils import process_vision_info
            from transformers import Qwen2VLForConditionalGeneration
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir)
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            self.tokenizer=None
            
        elif "llava" in self.model_path:
            import sys
            sys.path.append(os.path.join("vlmdrive"))
            from llava.model.builder import load_pretrained_model
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
            from llava.utils import disable_torch_init
            from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
            from llava.conversation import conv_templates

            disable_torch_init()
            self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model("liuhaotian/llava-v1.6-mistral-7b", None, "llava-v1.6-mistral-7b")
            self.image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            
        else:
            self.model = None
            self.processor = None
            self.tokenizer=None


    
    # ======================================================================================================
    #                                      Prompts (need further improvement)
    # ======================================================================================================
    
    def SceneDescription(self, obs_images, processor=None, model=None, tokenizer=None, prompt_template=None):

        key = "llava" if "llava" in self.model_path else "default"
        # prompt = """You are a autonomous driving labeller. You have access to these front-view camera images of a car. Imagine you are driving the car. Describe the driving scene according to traffic lights, other cars or pedestrians and lane markings."""

        # if "llava" in self.model_path:
        #     prompt = """You are an autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Provide a concise description of the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

        prompt = prompt_template["scene_prompt_template"][key]
        result = self.vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer)
        return result

    def DescribeObjects(self, obs_images, processor=None, model=None, tokenizer=None, prompt_template=None):

        
        # prompt = """You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

        prompt = prompt_template["object_prompt_template"]["default"]
        result = self.vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer)

        return result

    # def DescribeOrUpdateIntent(self, obs_images, prev_intent=None, processor=None, model=None, tokenizer=None):

    #     if prev_intent is None:
    #         prompt = """You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle. Imagine you are driving the car. Based on the lane markings and other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""

    #         if "llava" in self.model_path:
    #             prompt = """You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, provide a concise description of the desired intent of  the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""
            
    #     else:
    #         prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: """

    #         if "llava" in self.model_path:
    #             prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Provide a concise description explanation of your current intent: """

    #     result = self.vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer)

    #     return result


    def _get_ego_history(self, perception_memory_bank, agent_idx):
        """
        Builds a JSON string containing the ego vehicle's historical speed and curvature
        for each time frame in the perception_memory_bank.
        This JSON is eventually embedded in the final prompt.

        Returns:
            str: a JSON string with the structure:
                {
                "ego_history": [
                    {"timestamp": ..., "speed": ..., "curvature": ...},
                    ...
                ]
                }
        """

        ego_history_list = []  # We'll store each record as a dict

        prev_position = None
        prev_yaw = None
        dt = 0.5  # Example time interval (0.5s) if not specified

        for frame_data in perception_memory_bank:
            timestamp = frame_data['timestamp']
            # ego_pose shape: (3,) -> [x, y, yaw]
            ego_pose = frame_data['detmap_pose'][agent_idx].cpu().numpy()
            x, y, _ = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])
            yaw = frame_data['ego_yaw'][agent_idx]

            # Speed: approximate derivative
            if prev_position is not None:
                dx = x - prev_position[0]
                dy = y - prev_position[1]
                speed = np.sqrt(dx*dx + dy*dy) / dt
            else:
                speed = 0.0

            # Curvature: approximate derivative of yaw w.r.t. distance
            if prev_yaw is not None:
                d_yaw = yaw - prev_yaw
                distance = max(1e-6, speed * dt)
                curvature = d_yaw / distance
            else:
                curvature = 0.0

            record = {
                "timestamp": timestamp,
                "speed": round(speed, 3),
                "curvature": round(curvature, 5),
            }
            ego_history_list.append(record)

            prev_position = (x, y)
            prev_yaw = yaw

        # Wrap the list into a JSON structure
        # This is the JSON snippet that we can embed in the final prompt
        ego_history_json = {
            "ego_history": ego_history_list
        }

        # Convert to string. We can indent=2 for readability, or just one-liner
        prompt_string = json.dumps(ego_history_json, indent=2)
        return prompt_string


    def _gen_individual_waypoints(self, image, ego_history_prompt, model_config, perception_memory_bank, agent_idx):
        """
        Generates future speed-curvature pairs in a strict JSON format.
        The final prompt instructs the model to produce a JSON object with a specific key
        and shape. We embed the historical JSON info and optional scene/object descriptions.

        Returns:
            (result_str, scene_description, object_description)
            Where result_str is the raw output from the model (which ideally is also JSON),
            scene_description and object_description are strings from the sub-prompts.
        """

        # 1) Obtain scene description
        scene_description = self.SceneDescription(
            image,
            processor=self.processor,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_template=model_config["planning"]["prompt_template"]
        )
        # 2) Obtain object description
        object_description = self.DescribeObjects(
            image,
            processor=self.processor,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_template=model_config["planning"]["prompt_template"]
        )
        
        # 3) target_waypoint:
        target_waypoint = perception_memory_bank[-1]["target"][agent_idx]
        
        comb_prompt = \
            model_config["planning"]["prompt_template"]["comb_prompt"]["default"]\
                .format(scene_description=scene_description, 
                        object_description=object_description,
                        ego_history_prompt=ego_history_prompt, 
                        target_waypoint=target_waypoint.tolist())

        sys_message = model_config["planning"]["prompt_template"]["sys_message"]

        # Attempt inference, similar retry logic if necessary
        result = ""
        for _ in range(3):
            result = self.vlm_inference(
                text=comb_prompt,
                images=image,
                sys_message=sys_message,
                processor=self.processor,
                model=self.model,
                tokenizer=self.tokenizer
            )
            # We can do a quick check if there's a pattern "predicted_speeds_curvatures": [
            # if '"predicted_speeds_curvatures":' in result:
            #     break
            try:
                result = self._postprocess_result(result)
                break
            except:
                continue
            
        

        return result, scene_description, object_description
        

    def getMessage(self, prompt, image=None, model_path=None):
        if "llama" in model_path:
            message = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
        elif "qwen" in model_path:
            message = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]   
        return message

    
    def vlm_inference(self, text=None, images=None, sys_message=None, processor=None, model=None, tokenizer=None):
        
        if isinstance(images, np.ndarray):
            images = Image.fromarray(images)
            temp = f"debug/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}-buffer.png"
            images.save(temp)
            images = temp
        
        if  "llama" in self.model_path:
            from transformers import MllamaForConditionalGeneration 
            
            image = Image.open(images).convert('RGB')
            message = self.getMessage(text, model_path=self.model_path)
            input_text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            output = model.generate(**inputs, max_new_tokens=1024)

            output_text = processor.decode(output[0])

            if "llama" in self.model_path:
                output_text = re.findall(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', output_text, re.DOTALL)[0].strip()
            return output_text
        
        elif "qwen" in self.model_path:
            message = self.getMessage(text, image=images, model_path=self.model_path)
            text = processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(message) 
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

        elif "llava" in self.model_path:
            import sys
            sys.path.append(os.path.join("vlmdrive"))
            from llava.model.builder import load_pretrained_model
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
            from llava.utils import disable_torch_init
            from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
            from llava.conversation import conv_templates
            
            
            conv_mode = "mistral_instruct"
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN 
            if IMAGE_PLACEHOLDER in text: 
                if model.config.mm_use_im_start_end:
                    text = re.sub(IMAGE_PLACEHOLDER, image_token_se, text) 
                else:
                    text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, text) 
            else:
                if model.config.mm_use_im_start_end:
                    text = image_token_se + "\n" + text
                else:
                    text = DEFAULT_IMAGE_TOKEN + "\n" + text 

            conv = conv_templates[conv_mode].copy()  
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() 
            image = Image.open(images).convert('RGB')

            image_tensor = process_images([image], processor, model.config)[0] 

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True,
                    temperature=0.2,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=2048,
                    use_cache=True,
                    pad_token_id = tokenizer.eos_token_id,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs
                    
        elif "api" in self.model_path:
            API_BASE_URL = self.api_base_url
            API_KEY = self.api_key
            MODEL_NAME = self.api_model_name

            from openai import OpenAI

            import os
            # 设置环境变量来禁用代理
            os.environ['HTTPX_PROXIES'] = ''
            os.environ['no_proxy'] = '*'
            
            # 创建HTTPX客户端并显式设置代理为空
            try:
                http_client = httpx.Client(
                    proxies={},
                    transport=httpx.HTTPTransport(retries=3)
                )
            except:
                # For httpx >= 0.24.0
                http_client = httpx.Client(
                    transport=httpx.HTTPTransport(retries=3)
                )
            
            client = OpenAI(
                base_url=API_BASE_URL,
                api_key=API_KEY,
                http_client=http_client
            )
            
            # 处理图片路径
            if isinstance(images, str):
                images = [images]
            
            # 构建消息内容
            content = []
            # 添加文本内容
            content.append({"type": "text", "text": text})
            # 添加图片内容
            for img_path in images:
                if isinstance(img_path, str):
                    # 如果是本地文件，需要转换为 base64
                    if os.path.exists(img_path):
                        with open(img_path, "rb") as image_file:
                            import base64
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            })
            
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            if sys_message is not None:
                sys_message_dict = {
                    "role": "system",
                    "content": sys_message
                }
                PROMPT_MESSAGES.append(sys_message_dict)
            params = {
                "model": MODEL_NAME,
                "messages": PROMPT_MESSAGES,
                "max_tokens": 2048,
            }

            result = client.chat.completions.create(**params)

            print(f"================================\n{result.choices[0].message.content}")
            
            return result.choices[0].message.content
            
        else:
            raise ValueError(f"Unsupported model path: {self.model_path}")

    def _get_ego_front_image(self, perception_memory_bank, agent_idx):
        """Get the front image of the ego vehicle

        Args:
            perception_memory_bank (List[dict]): [car0, car1, ...] each element is a dictionary of perception record
        """

        return perception_memory_bank[-1]['rgb_front'][agent_idx] # (H, W, 3)
    
    def _get_all_objects(self, perception_memory_bank):
        """Get all detected objects including bbox from CAV

        Args:
            perception_memory_bank (List[dict]): [car0, car1, ...]
        """
        bboxes = perception_memory_bank[-1]["object_list"]
        return bboxes
        

    def _postprocess_result(self, result, init_x=0.0, init_y=0.0, init_yaw=0.0, dt=0.5):
        """
        Parse the predicted JSON from the LLM output, which should contain a key
        'predicted_speeds_curvatures' with a list of [speed, curvature] pairs.
        Then, starting from an initial pose (init_x, init_y, init_yaw), we integrate
        these speeds and curvatures over time interval dt to produce a series
        of (x, y) waypoints.

        Args:
            result (str): The raw string output from the LLM, presumably including JSON.
            init_x (float): Initial x position of the ego vehicle.
            init_y (float): Initial y position of the ego vehicle.
            init_yaw (float): Initial yaw (heading) of the ego vehicle, in radians.
            dt (float): Time increment per speed-curvature pair (e.g., 0.5s).

        Returns:
            torch.Tensor: A tensor of shape (N, 2), where N is the number of predicted
                        speed-curvature steps, each row is [x, y].
        """

        # 1) Extract JSON substring from the result using regex
        json_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```|\{[\s\S]*\}", re.MULTILINE)
        json_str = None
        try:
            json_match = json_pattern.search(result)
            if json_match:
                # If the group(1) is not empty, we use that; otherwise we use the entire match
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            else:
                raise ValueError("No JSON content found in the model output.")
        except Exception as e:
            raise ValueError(f"Failed to locate JSON via regex: {str(e)}")

        # 2) Load the JSON structure
        try:
            json_result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        # 3) Retrieve the [speed, curvature] pairs from the JSON
        if "predicted_speeds_curvatures" not in json_result:
            raise ValueError("JSON does not contain 'predicted_speeds_curvatures' key.")

        speed_curv_list = json_result["predicted_speeds_curvatures"]
        if not isinstance(speed_curv_list, list):
            raise ValueError("The 'predicted_speeds_curvatures' should be a list of pairs.")

        # 4) Integrate each (speed, curvature) pair to produce (x, y) waypoints.
        x, y, yaw = init_x, init_y, init_yaw
        waypoints = []

        for i, pair in enumerate(speed_curv_list):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Expected each item to be [speed, curvature], got {pair} at index {i}.")

            speed, curvature = float(pair[0]), float(pair[1])
            # Distance traveled over dt
            dist = speed * dt

            # Yaw change from curvature * distance
            # This is a simple approach; advanced methods could use more precise arc integration
            yaw_new = yaw + curvature * dist

            # Approximate the new (x, y) based on the updated heading
            # One naive approach is to use the new yaw for direction:
            x_new = x + dist * math.cos(yaw_new)
            y_new = y + dist * math.sin(yaw_new)

            # Update for the next iteration
            x, y, yaw = x_new, y_new, yaw_new
            waypoints.append([x, y])

        # 5) Convert to torch.Tensor and return
        waypoints = np.array(waypoints, dtype=np.float32)  # shape (N, 2)
        return torch.from_numpy(waypoints)

    def forward(self, perception_memory_bank, model_config) -> torch.Tensor:

        N = perception_memory_bank[-1]["rgb_front"].shape[0]
        waypoints = []

        for i in range(N):
            # (1) retrieve latest captured front images
            front_image = \
                self._get_ego_front_image(perception_memory_bank, i) # (H, W, 3)
            # (2) prepare promt
            # import pdb; pdb.set_trace()
            ego_history_prompt = \
                self._get_ego_history(perception_memory_bank, i)
            # (3) generate waypoints
            pred_waypoints, _, _ = \
                self._gen_individual_waypoints(front_image, 
                                               ego_history_prompt, 
                                               model_config,
                                               perception_memory_bank,
                                               i)
            # (4) postprocess waypoints
            # i_car_waypoints = self._postprocess_result(pred_waypoints)
            
            waypoints.append(pred_waypoints)

        waypoints = torch.stack(waypoints)

        return waypoints