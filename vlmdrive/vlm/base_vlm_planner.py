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
import re
import argparse
from datetime import datetime
from math import atan2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from pyquaternion import Quaternion
from scipy.integrate import cumulative_trapezoid

import json
from vlmdrive.utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN


@VLMDRIVE_REGISTRY.register
class BaseVLMWaypointPlanner(nn.Module):
    """
    WaypointPlanner language prompt
    """
    
    
    def __init__(self, name, **kwargs):
        super().__init__()
        self.model_path = name
        
        # TODO: More VLMs and VLLM quantization
        
        if "llama" in self.model_path:
            from transformers import MllamaForConditionalGeneration 
            
            model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.tokenizer=None
            
        elif "qwen" in self.model_path:
            from qwen_vl_utils import process_vision_info
            from transformers import Qwen2VLForConditionalGeneration
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
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
    
    
    def SceneDescription(self, obs_images, processor=None, model=None, tokenizer=None, args=None):
        prompt = f"""You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

        if "llava" in args.model_path:
            prompt = f"""You are an autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Provide a concise description of the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

        result = self.vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
        return result

    def DescribeObjects(self, obs_images, processor=None, model=None, tokenizer=None, args=None):

        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

        result = self.vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)

        return result

    def DescribeOrUpdateIntent(self, obs_images, prev_intent=None, processor=None, model=None, tokenizer=None, args=None):

        if prev_intent is None:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""

            if "llava" in args.model_path:
                prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, provide a concise description of the desired intent of  the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""
            
        else:
            prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: """

            if "llava" in args.model_path:
                prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Provide a concise description explanation of your current intent: """

        result = self.vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)

        return result
    
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

    
    def vlm_inference(self, text=None, images=None, sys_message=None, processor=None, model=None, tokenizer=None, model_path=None):
        
        if isinstance(images, np.ndarray):
            images = Image.fromarray(images)
            temp = "debug/buffer.png"
            images.save(temp)
            images = temp
        
        if  "llama" in model_path:
            from transformers import MllamaForConditionalGeneration 
            
            image = Image.open(images).convert('RGB')
            message = self.getMessage(text, model_path=model_path)
            input_text = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            output = model.generate(**inputs, max_new_tokens=2048)

            output_text = processor.decode(output[0])

            if "llama" in model_path:
                output_text = re.findall(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', output_text, re.DOTALL)[0].strip()
            return output_text
        
        elif "qwen" in model_path:
            message = self.getMessage(text, image=images, model_path=model_path)
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
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]

        elif "llava" in model_path:
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
                    
        elif "gpt" in model_path:
            
            from openai import OpenAI
            client = OpenAI(api_key="[your-openai-api-key]")
            
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        *map(lambda x: {"image": x, "resize": 768}, images),
                        text,
                    ],
                },
            ]
            if sys_message is not None:
                sys_message_dict = {
                    "role": "system",
                    "content": sys_message
                }
                PROMPT_MESSAGES.append(sys_message_dict)
            params = {
                "model": "gpt-4o-2024-11-20",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 400,
            }

            result = client.chat.completions.create(**params)
            
            return result.choices[0].message.content
            
        else:
            raise ValueError(f"Unsupported model path: {model_path}")

            
        
        
    def GenerateMotion(self, obs_images, obs_waypoints, obs_velocities, obs_curvatures, given_intent, processor=None, model=None, tokenizer=None, args=None):
        # assert len(obs_images) == len(obs_waypoints)

        scene_description, object_description, intent_description = None, None, None

        if args.method == "openemma":
            scene_description = self.SceneDescription(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
            object_description = self.DescribeObjects(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
            intent_description = self.DescribeOrUpdateIntent(obs_images, prev_intent=given_intent, processor=processor, model=model, tokenizer=tokenizer, args=args)
            print(f'Scene Description: {scene_description}')
            print(f'Object Description: {object_description}')
            print(f'Intent Description: {intent_description}')

        # Convert array waypoints to string.
        obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
        obs_waypoints_str = ", ".join(obs_waypoints_str)
        obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
        obs_curvatures = obs_curvatures * 100
        obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
        obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

        
        print(f'Observed Speed and Curvature: {obs_speed_curvature_str}')

        sys_message = ("You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.\n")

        if args.method == "openemma":
            prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
            The scene is described as follows: {scene_description}. 
            The identified critical objects are {object_description}. 
            The car's intent is {intent_description}. 
            The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
            Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
        else:
            prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
            The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
            Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
        for rho in range(3):
            result = self.vlm_inference(text=prompt, images=obs_images, sys_message=sys_message, processor=processor, model=model, tokenizer=tokenizer, args=args)
            if not "unable" in result and not "sorry" in result and "[" in result:
                break
        return result, scene_description, object_description, intent_description


    def forward(self, perception_memory_bank) -> torch.Tensor:
        
        def generate_waypoints(curr, target, speed=5.0, max_steps=10):
            direction = target - curr[:, :2]
            dist = torch.norm(direction, dim=1, keepdim=True)
            steps = min(int(torch.ceil(dist / speed).item()), max_steps)
            waypoints = [curr[:, :2] + direction / dist * speed * (i + 1) for i in range(steps)]
            return torch.cat(waypoints + [target] * (max_steps - steps))
        
        waypoints_agents = []
        for i, frame_data in enumerate(perception_memory_bank):
            ego_positions = frame_data['detmap_pose']
            destinations = frame_data['target']
            waypoints = generate_waypoints(ego_positions, destinations)
            waypoints_agents.append(waypoints)
        waypoints_agents = torch.stack(waypoints_agents) # [N, 10, 2]
            
        
        # # TODO: Uncomment and Implement the following
        # waypoints_agents = []
        # for agent_idx in range(len(perception_memory_bank)):
            
        #     vlm_prompt = self.generate_vlm_prompt(perception_memory_bank, agent_idx)
        #     image = self.get_image(perception_memory_bank[agent_idx])
        #     result = self.vlm_inference(text=vlm_prompt, images=image, sys_message="", processor=self.processor, model=self.model, tokenizer=self.tokenizer, model_path=self.model_path)
        #     result_postprocessed = self.postprocess_result(result)
        #     waypoints_agents.append(result_postprocessed) # [N, 10, 2]
            
        return waypoints_agents
        
        
    def get_image(self, perception_memory_bank):
        # TODO: Implement this function
        return [perception_memory_bank[i]['rgb_front'] for i in range(len(perception_memory_bank))]
    
    def postprocess_result(self, result):
        # TODO: Implement this function
        
        return result
        
    def generate_vlm_prompt(self, perception_memory_bank, agent_idx):
        # TODO: Implement this function

        # Header / high-level instructions
        prompt_lines = [
            "Information Provided:",
            "1. Historical positions (5 frames) of the ego agents.",
            "2. Historical bounding boxes of detected objects (same 5 frames).",
            "3. Destination of each ego agent.",
            "",
            "Goal:",
            "Predict 20 future waypoints for each ego agent based on the above information.",
            "",
            "Answer Format:",
            "Please output a JSON-like structure with the key `predicted_waypoints` containing a list of 20 (x, y) pairs for each ego agent. For example:",
            "{",
            '  "predicted_waypoints": [',
            "     [x1, y1],",
            "     [x2, y2],",
            "     ... up to 20 waypoints ...",
            "   ]",
            "}",
            "",
            "=======================================",
            "Data Table (Last 5 Frames):",
            ""
        ]

        # Table headers
        table_header = "| Frame | Ego Positions (x, y, yaw) | Object BBoxes [(x1,y1),(x2,y2),...] | Destination (x, y) |"
        table_separator = "|---|---|---|---|"
        prompt_lines.append(table_header)
        prompt_lines.append(table_separator)

        # Build table rows
        for i, frame_data in enumerate(perception_memory_bank):
            # ego poses (N agents). Suppose you have N=1 for a single agent. Adjust if you have multiple agents.
            ego_positions = np.round(frame_data['detmap_pose'].cpu().numpy(), 2).tolist() # shape (N, 3)
            object_bboxes = np.round(frame_data['object_list'].cpu().numpy(), 2).tolist() # shape (N, K, 4, 2) or similar
            destinations = np.round(frame_data['target'].cpu().numpy(), 2).tolist()		
            # Format each row. This example assumes N=1 for simplicity:
            row_ego_pose = str(ego_positions)
            row_bboxes = str(object_bboxes)
            row_dest = str(destinations)

            prompt_lines.append(
                f"| {i} | {row_ego_pose} | {row_bboxes} | {row_dest} |"
            )

        # Join everything into a single prompt string
        prompt = "\n".join(prompt_lines)
        return prompt


