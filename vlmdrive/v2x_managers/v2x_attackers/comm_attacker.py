import collections
from copy import deepcopy
from vlmdrive.v2x_managers.v2x_attackers.base_attacker import BaseAttacker, sub_attack
import random
import time
import os
import numpy as np
import attack_counter
import re


class JammingAttacker(BaseAttacker):
    
    ATT_TYPE = "Jamming Attack"
    
    @sub_attack
    def message_loss_attack(self, message_item, self_message, ego_idx):
        
        attack_counter.attack_counter.tick() #counter for the start of the attack
        attack_counter.attack_counter.set_message("Message Loss Attack") #message to print on in the image
        
        if attack_counter.attack_counter.start_attack():
            print(f"Executing jamming attack ...")
            return None
        else:
            return message_item
        
    # # @sub_attack
    # def llm_denial_of_service(self, message, ego_idx):
    #     attack_counter.attack_counter.tick() #counter for the start of the attack
    #     attack_counter.attack_counter.set_message("LLM Denial of Service") #message to print on in the image
    #     if attack_counter.attack_counter.start_attack():
    #         print(f"Executing llm denial of service attack ...")
    #         # TODO: Implement logic to flood or block communication
    #         # Could duplicate messages or inject noise
    #         # print(message)
    #         # print(ego_idx)
    #         # import pdb; pdb.set_trace()
    #         if ego_idx == 2:
    #             message = message * 100
    #         return message
    #     else:
    #         return message
        
        
class ReplayAttacker(BaseAttacker):
    
    ATT_TYPE = "Replay Attack"
    
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
        
        
class SybilAttacker(BaseAttacker):
    
    def attack(self, collab_agent_message_collected, self_message, ego_idx, spacing=5.0, yaw_std = 0.5):
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

    
