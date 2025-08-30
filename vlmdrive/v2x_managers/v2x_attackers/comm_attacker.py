import collections
from copy import deepcopy
from vlmdrive.v2x_managers.v2x_attackers.base_attacker import BaseAttacker, sub_attack
import random
import time
import os
import numpy as np
import attack_counter
import re


class CommAttacker(BaseAttacker):
    
    ATT_TYPE = "Communication Attack"
    
    def __init__(self, *args, **kwargs):
        self.replay_buffer = collections.deque(maxlen = 15)
        super().__init__(*args, **kwargs)
    
    
    @sub_attack
    def spoofing_attack(self, message_item, self_message, ego_idx, noise_std=5.0, yaw_noise_std = 1):
        
        attack_counter.attack_counter.tick() #counter for the start of the attack
        attack_counter.attack_counter.set_message("Spoofing Attack") #message to print on in the image
        if attack_counter.attack_counter.start_attack():
            print(f"Executing spoofing attack ...")
            msg = deepcopy(message_item) # only attack the first message
            position = msg["position"] #extract position information
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", position) #extract number in position information
            position_val = np.array(nums, dtype=float) # convert the number into np array
            speed = float(msg ["speed"]) #extract speed information
            yaw = float(msg["ego_yaw"]) #extract yaw information

            position_noise = np.random.normal(0, noise_std, size=position_val.shape)
            noised_position = position_val + position_noise # add noise to the position
            noised_speed = speed + np.random.normal(loc=0.0, scale=1.0) # add noise to the speed
            yaw_noise = np.random.normal(0, yaw_noise_std) 
            noised_yaw = yaw + yaw_noise # add noise to the yaw

            #put these modified information back
            msg["position"] = re.sub(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lambda m, it=iter(noised_position ): f"{next(it):.5f}", msg["position"])
            msg["speed"] = noised_speed
            msg["ego_yaw"] = noised_yaw
            return msg
        else:
            return message_item

    
    @sub_attack
    def message_loss_attack(self, message_item, self_message, ego_idx):
        
        attack_counter.attack_counter.tick() #counter for the start of the attack
        attack_counter.attack_counter.set_message("Message Loss Attack") #message to print on in the image
        
        if attack_counter.attack_counter.start_attack():
            print(f"Executing message loss attack ...")
            return None
        else:
            return message_item
        
    # @sub_attack
    def llm_denial_of_service(self, message, ego_idx):
        attack_counter.attack_counter.tick() #counter for the start of the attack
        attack_counter.attack_counter.set_message("LLM Denial of Service") #message to print on in the image
        if attack_counter.attack_counter.start_attack():
            print(f"Executing llm denial of service attack ...")
            # TODO: Implement logic to flood or block communication
            # Could duplicate messages or inject noise
            # print(message)
            # print(ego_idx)
            # import pdb; pdb.set_trace()
            if ego_idx == 2:
                message = message * 100
            return message
        else:
            return message
        
    @sub_attack
    def replay_attack(self, message_item, self_message, ego_idx):
        attack_counter.attack_counter.tick()  # counter for the start of the attack
        attack_counter.attack_counter.set_message("Replay Attack")  # message to print in the image
        if attack_counter.attack_counter.start_attack():
            print(f"Executing replay attack ...")
            buf = self.message_buffer.get(message_item["idx"])
            if buf and len(buf) > 0:
                return deepcopy(buf[0])  # earliest buffered message from this agent
            return message_item
        else:
            return message_item