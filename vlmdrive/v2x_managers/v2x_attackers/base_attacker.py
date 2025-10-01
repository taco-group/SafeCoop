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
        self.sybil_num = kwargs.get("sybil_num", 3)
        
    def _buffer_message(self, message, ego_idx):
        self.message_buffer[ego_idx].append(deepcopy(message))
    
    def _log_main_category(self):
        """
        Log the main attack category.
        """
        print(f"Attack Main Type: {self.ATT_TYPE}")
        attack_counter.attack_counter.tick()  # counter for the start of the attack
        attack_counter.attack_counter.set_message(self.ATT_TYPE)  # message to print in the image
        
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
    