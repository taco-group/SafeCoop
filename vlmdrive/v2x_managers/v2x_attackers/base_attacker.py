from abc import ABC, abstractmethod
from copy import deepcopy
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
        self.atker = kwargs.get('atker', None)
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
        
    def attack(self, message, ego_idx, **kwargs):
        """
        Simulate a perceptual attack on the message.
        """
        self._log_main_category()
        message = deepcopy(message)
        attacked_message = []
        
        for message_id, message_item in enumerate(message):
            if message_item['idx'] == ego_idx:
                # Do not attack the ego message. We assume it to be benign.
                continue
            self._buffer_message(message_item, message_item['idx'])
            att_method = self._choose_sub_method()
            message_item = att_method(message_item, **kwargs)
            # Update the message item with the attacked version 
            # (att_method supposed to modify the item in place, an replacement is add in case it returns a new item)
            attacked_message.append(message_item)
        
        return attacked_message
