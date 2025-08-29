import collections
from vlmdrive.v2x_managers.v2x_attackers.base_attacker import BaseAttacker, sub_attack
import random
import time
import os
import numpy as np
import attack_counter
import re


class CommAttacker(BaseAttacker):
    
    ATT_TYPE = "Communication Attack"
    
    def __init__(self, *args, atker_ids=None, **kwargs):
        self.replay_buffer = collections.deque(maxlen = 15)
        self.atker_ids = list(atker_ids) if atker_ids is not None else []

        super().__init__(*args, **kwargs)
    
    
    @sub_attack
    def sybil_attack(self, message, ego_idx, num_fake_vehicle = 3, spacing=5.0, yaw_std = 0.5):
        if ego_idx == self.atker_ids:
            return message
        
        attack_counter.attack_counter.tick() #counter for the start of the attack
        attack_counter.attack_counter.set_message("Sybil Attack") #message to print on in the image

        
        if attack_counter.attack_counter.start_attack():
            print(f"Executing sybil attack...")
            sybil = []
            # extract information from the real message
            position = message[0]["position"]
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", position)
            speed = float(message[0]["speed"])
            scene_description = message[0]["scene_description"]
            intent_description = message[0]["intent_description"]
            target_description = message[0]["target_description"]
            object_description = message[0]["object_description"]
            
            #creating fake vehicle based on the data extracted
            for i in range (num_fake_vehicle - 1, -1, -1):
                yaw =  np.random.normal(0, yaw_std)
                direction = np.array([np.cos(yaw), np.sin(yaw)]) 
                position_val = np.array(nums, dtype=float)
                pos = position_val + i * spacing * direction
                new_speed = speed + np.random.normal(loc=0.0, scale=1.0)
                fake = {
                "idx": 4 + i,
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
                # import pdb; pdb.set_trace()
            return message + sybil
        else:
            return message
    
    
    
    
    
    @sub_attack
    def spoofing_attack(self, message, ego_idx, noise_std=1.0, yaw_noise_std = 0.5):
        if ego_idx == self.atker_ids:
            return message
        attack_counter.attack_counter.tick() #counter for the start of the attack
        attack_counter.attack_counter.set_message("Spoofing Attack") #message to print on in the image
        if attack_counter.attack_counter.start_attack():
            print(f"Executing spoofing attack ...")
            spoofed_message= []
            for m in message:
                msg_copy = m.copy()
                # print(m)
                if m["idx"] == self.atker_ids:
                    position = msg_copy ["position"] #extract position information
                    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", position) #extract number in position information
                    position_val = np.array(nums, dtype=float) # convert the number into np array
                    speed = float(msg_copy ["speed"]) #extract speed information
                    yaw = float(msg_copy["ego_yaw"]) #extract yaw information

                    position_noise = np.random.normal(0, noise_std, size=position_val.shape)
                    noised_position = position_val + position_noise # add noise to the position
                    noised_speed = speed + np.random.normal(loc=0.0, scale=1.0) # add noise to the speed
                    yaw_noise = np.random.normal(0, yaw_noise_std) 
                    noised_yaw = yaw + yaw_noise # add noise to the yaw

                    #put these modified information back
                    msg_copy["position"] = re.sub(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lambda m, it=iter(noised_position ): f"{next(it):.5f}", msg_copy["position"])
                    msg_copy["speed"] = noised_speed
                    msg_copy["ego_yaw"] = noised_yaw
                
                spoofed_message.append(msg_copy)
            # import pdb; pdb.set_trace()

            return spoofed_message
        else:
            return message

    
    @sub_attack
    def message_loss_attack(self, message, ego_idx):
        if ego_idx == self.atker_ids:
            return message
        
        attack_counter.attack_counter.tick() #counter for the start of the attack
        attack_counter.attack_counter.set_message("Message Loss Attack") #message to print on in the image
        
        if attack_counter.attack_counter.start_attack():
            print(f"Executing message loss attack ...")
            new_message = []
            for m in message:
                if m["idx"] != self.atker_ids: # only keep the message that is not from the attacker
                    new_message.append(m)
            return new_message
        else:
            return message
        
    @sub_attack
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
    def replay_attack(self, message, ego_idx):
        if ego_idx == self.atker_ids:
            return message

        attack_counter.attack_counter.tick() #counter for the start of the attack
        attack_counter.attack_counter.set_message("Replay Attack") #message to print on in the image
        new_message = []
        for m in message:
            if m["idx"] == self.atker_ids:
                self.replay_buffer.append(m) # store current data
                
                if attack_counter.attack_counter.start_attack(): # start to attack
                    new_message.append(self.replay_buffer.popleft()) # ues the left most data we stored in the queue
            
            else:
                if attack_counter.attack_counter.start_attack():
                    new_message.append(m)

        return new_message if new_message else message