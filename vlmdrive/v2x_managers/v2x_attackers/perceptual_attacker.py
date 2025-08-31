from vlmdrive.v2x_managers.v2x_attackers.base_attacker import BaseAttacker, sub_attack
import random


class PerceptualAttacker(BaseAttacker):
    
    ATT_TYPE = "Perceptual Attack"

    def _choose_sub_method(self):
        """
        Choose the hallucination sub-attack for development purpose.
        """
        return self.hallucination

    # @sub_attack
    def omission(self, message):
        print(f"Omitting info ...")
        
        # TODO: Implement the logic to omit information from the message.
        return message

    @sub_attack
    def hallucination(self, message_item, self_message, ego_idx):
        """
        This is the main hallucination attack method.
        It should inject the hallucination into the any one of the collab agents message.
        It has 2 stages:
            1. Hazard identification:
                - Identify the potentialhazard in the victim car's view. 
                - Fake the hazard as it has already happened and inject the hazard description into the object description of the first collab agent message.
        """
        print("Adding hallucination ...")

        tpl_group = self.prompt_template.get("hallucination", {})
        # Build self driving scene description from self_message
        self_scene = self_message.get("scene_description", "") if self_message else ""
        self_objects = self_message.get("object_description", "") if self_message else ""
        self_intent = self_message.get("intent_description", "") if self_message else ""
        self_driving_scene_description = (
            f"Scene description: {self_scene}\nObjects description: {self_objects}\nIntent description: {self_intent}"
        )

        # Stage 1: hazard identification
        stage1_t = tpl_group.get("hazard_identification_prompt", "")
        stage1_prompt = stage1_t.format(self_driving_scene_description=self_driving_scene_description)
        print("Stage1 prompt:", stage1_prompt)       
        stage1_out = self.atker.infer(text=stage1_prompt)
        print("Stage1 output:", stage1_out)

        attacker_agent = message_item
        # get existing object_description safely
        orig = attacker_agent.get("object_description", "")
        # ensure output is a string
        injected = str(stage1_out or "")
        # safe assignment (avoid += on missing/None)
        attacker_agent["object_description"] = (orig + "\n" + injected).strip() if orig else injected.strip()
        print("Attacker agent object description:", attacker_agent["object_description"])

        return message_item
    
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
    
    # @sub_attack
    def mislabeling(self, message):
        print(f"Mislabeling info ...")
        # TODO: Implement the logic to mislabel information in the message.
        return message

    # @sub_attack
    def localization_error(self, message):
        print(f"Introducing localization error ...")
        # TODO: Implement the logic to introduce localization error in the message.
        return message
        
    # @sub_attack
    def temporal_error(self, message): # Treated as communication attack.
        print(f"Introducing temporal error in ...")
        # TODO: Implement the logic to introduce temporal error in the message.
        # Note: Skip this for now since the required image message is not provided in the message.
        return message
        
    # @sub_attack
    def SL_misalignment(self, message):
        print(f"Introducing sensor-to-language misalignment in ...")
        # TODO: Implement the logic to introduce sensor-to-language misalignment in the message.
        # Note: Skip this for now since the required image message is not provided in the message.
        return message
   
   
   
   
   