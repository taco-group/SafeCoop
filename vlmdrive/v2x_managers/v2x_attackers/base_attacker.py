from abc import ABC, abstractmethod
from copy import deepcopy
import random

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
        self.prompt_template = kwargs.get("prompt_template", {})
    
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
        message = deepcopy(collab_agent_message_collected)
        attacked_message = []
        
        
        att_method = self._choose_sub_method()
        
        # Update the message with the attacked version 
        attacked_message = att_method(message, self_message, ego_idx)
        
        return attacked_message
