from abc import ABC, abstractmethod
from copy import deepcopy
import re
import numpy as np
import attack_counter
import random
from collections import deque, defaultdict

sub_attack_methods_registry = {}

def sub_attack(func):
    sub_attack_methods_registry.setdefault(func.__qualname__.split('.')[0], []).append(func)
    return func

class BaseAttacker(ABC):
    
    def __init__(self, *args, **kwargs):
        cls_name = self.__class__.__name__
        funcs = sub_attack_methods_registry.get(cls_name, [])
        self.sub_attack_methods = [func.__get__(self) for func in funcs]
        self.atker_ids = list(kwargs.get('atker_ids', []))
        self.atker = kwargs.get('atker', None)
        self.ego_num = kwargs.get('ego_num', 10)
        self.message_buffer_size = kwargs.get('message_buffer_size', 20)
        """
        {
            atker_id_1: deque([message1, message2, ...]),
            atker_id_2: deque([message1, message2, ...]),
            ... 
        }
        """
        self.message_buffer = defaultdict(
            lambda: deque(maxlen=self.message_buffer_size)
        )
        self.image_placeholder = kwargs.get('IMAGE_PLACEHOLDER', '<IMAGE_PLACEHOLDER>')
        self.prompt_template = kwargs.get("prompt_template", {})
        self.with_sybil = kwargs.get("with_sybil", False)
        self.sybil_num = kwargs.get("sybil_num", 3)
        
    def _buffer_message(self, message, ego_idx):
        self.message_buffer[ego_idx].append(deepcopy(message))
    
    def _log_main_category(self):
        """
        Log the main attack category.
        """
        print(f"Attack Main Type: {self.ATT_TYPE}")
        
    def _choose_sub_method(self):
        """
        Choose a sub attack category.
        """
        if not self.sub_attack_methods:
            raise ValueError("No sub attack methods available.")
        sub_att_method = random.choice(self.sub_attack_methods)
        self._log_sub_method(sub_att_method)
        return sub_att_method
        
    def _log_sub_method(self, sub_att_method):
        """
        Log the sub attack method.
        """
        print(f"Sub Attack Type: {sub_att_method.__name__}")
        
    def attack(self, collab_agent_message_collected, self_message, ego_idx):
        """
        Simulate a perceptual attack on the message.
        """
        self._log_main_category()
        for msg in collab_agent_message_collected:
            self.message_buffer[msg['idx']].append(deepcopy(msg))
        message = deepcopy(collab_agent_message_collected)
        attacked_message = []
        for message_item in message:
            message_id = message_item['idx']
            if message_id not in self.atker_ids:
                attacked_message.append(message_item)
                continue
            self._buffer_message(message_item, message_id)
            att_method = self._choose_sub_method()
            message_item = att_method(message_item, self_message, ego_idx)
            if message_item:
                attacked_message.append(message_item)
        return attacked_message
    
    def sybil_attack(self, collab_agent_message_collected, self_message, ego_idx, spacing=5.0, yaw_std = 0.5):
        attack_counter.attack_counter.tick() #counter for the start of the attack
        num_fake_vehicle = self.sybil_num
        attack_counter.attack_counter.set_message(f"(Sybil - count={num_fake_vehicle})") #message to print on in the image

        cur_idx = 0
        sybil = []
        for message_item in collab_agent_message_collected:
            if message_item['idx'] not in self.atker_ids:
                continue
            if attack_counter.attack_counter.start_attack():
                print(f"Executing sybil attack...")
                # extract information from the real message
                position = message_item["position"]
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", position)
                speed = float(message_item["speed"])
                scene_description = message_item["scene_description"]
                intent_description = message_item["intent_description"]
                target_description = message_item["target_description"]
                object_description = message_item["object_description"]
                #creating fake vehicle based on the data extracted
                for i in range(num_fake_vehicle):
                    yaw =  np.random.normal(0, yaw_std)
                    direction = np.array([np.cos(yaw), np.sin(yaw)]) 
                    position_val = np.array(nums, dtype=float)
                    pos = position_val + i * spacing * direction
                    new_speed = speed + np.random.normal(loc=0.0, scale=1.0)
                    fake = {
                    "idx": self.ego_num + cur_idx,
                    "position": re.sub(
                        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                        lambda m, it=iter(pos): f"{next(it):.5f}", position
                    ),
                    "speed": new_speed,
                    "ego_yaw": yaw,
                    "scene_description": scene_description,
                    "object_description": object_description + ' Car in the center, approaching the intersection, moving straight ahead', #add an extra vehicle in the object
                    "target_description": target_description,
                    "intent_description": intent_description
                    }
                    sybil.append(fake)
                    cur_idx += 1
        return collab_agent_message_collected + sybil

    
