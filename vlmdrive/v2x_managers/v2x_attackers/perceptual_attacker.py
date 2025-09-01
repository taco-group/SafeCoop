from vlmdrive.v2x_managers.v2x_attackers.base_attacker import BaseAttacker, sub_attack
import random


class PerceptualAttacker(BaseAttacker):
    
    ATT_TYPE = "Perceptual Attack"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _choose_sub_method(self):
        """
        Choose the omission sub-attack for development purpose.
        """
        return self.omission

    @sub_attack
    def omission(self, collab_agent_message, self_message, ego_idx):
        """
        Omission attack: Randomly drop object descriptions from the attacker's perception
        to cause the attacker to miss important objects in the environment.
        """
        print("Adding omission attack...")
        
        if not collab_agent_message:
            return collab_agent_message
        
        # Select attacker agent (hardcoded to agent 0)
        attacker_agent = collab_agent_message[0]
        original_object_desc = attacker_agent.get("object_description", "")
        
        if not original_object_desc:
            print("No object description to omit from")
            return collab_agent_message
        
        # Step 1: Extract objects from the description
        extracted_objects = self._extract_objects_from_description(original_object_desc)
        
        if not extracted_objects:
            print("No objects found to omit")
            return collab_agent_message
        
        # Step 2: Randomly select objects to omit (30-70% of objects)
        num_objects = len(extracted_objects)
        min_to_omit = max(1, int(num_objects * 0.3))  # At least 30%
        max_to_omit = max(1, int(num_objects * 0.7))  # At most 70%
        num_to_omit = random.randint(min_to_omit, max_to_omit)
        import pdb; pdb.set_trace()
        objects_to_omit = random.sample(extracted_objects, min(num_to_omit, num_objects))
        
        # Step 3: Remove the selected objects from the description
        modified_description = self._remove_objects_from_description(original_object_desc, objects_to_omit)
        
        # Step 4: Update the attacker's message
        attacker_agent["object_description"] = modified_description
        
        omitted_labels = [obj['label'] for obj in objects_to_omit]
        print(f"Omitted objects: {omitted_labels}")
        
        return collab_agent_message

    @sub_attack
    def hallucination(self, collab_agent_message, self_message, ego_idx):
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

        # hard code to inject the hazard into the first collab agent message
        # TODO: Remove the hard code to randomly select the attacker agent
        attacker_agent = collab_agent_message[0]
        # get existing object_description safely
        orig = attacker_agent.get("object_description", "")
        # ensure output is a string
        injected = str(stage1_out or "")
        # safe assignment (avoid += on missing/None)
        attacker_agent["object_description"] = (orig + "\n" + injected).strip() if orig else injected.strip()
        print("Attacker agent object description:", attacker_agent["object_description"])

        return collab_agent_message
    
    @sub_attack
    def mislabeling(self, collab_agent_message, self_message, ego_idx):
        """
        Mislabelling attack: Randomly modify object labels in the attacker's perception
        to cause confusion and potentially dangerous driving decisions.
        """
        print("Adding mislabelling attack...")
        
        if not collab_agent_message:
            return collab_agent_message
        
        # Select attacker agent (currently hardcoded to first agent, can be randomized)
        attacker_agent = collab_agent_message[0]
        original_object_desc = attacker_agent.get("object_description", "")
        
        if not original_object_desc:
            print("No object description to mislabel")
            return collab_agent_message
        
        # Step 1: Extract objects from the description
        extracted_objects = self._extract_objects_from_description(original_object_desc)
        
        if not extracted_objects:
            print("No objects found to mislabel")
            return collab_agent_message
        
        # Step 2: Select objects to mislabel (random selection)
        objects_to_mislabel = self._select_objects_for_mislabelling(extracted_objects)
        import pdb; pdb.set_trace()
        # Step 3: Generate mislabellings for selected objects
        mislabelled_desc = self._apply_mislabellings(original_object_desc, objects_to_mislabel)
        
        # Step 4: Update the victim's message
        attacker_agent["object_description"] = mislabelled_desc
        print(f"Mislabelled objects: {[obj['label'] for obj in objects_to_mislabel]}")
        
        return collab_agent_message

    @sub_attack
    def localization_error(self, message):
        print(f"Introducing localization error ...")
        # TODO: Implement the logic to introduce localization error in the message.
        return message
        
    @sub_attack
    def temporal_error(self, message): # Treated as communication attack.
        print(f"Introducing temporal error in ...")
        # TODO: Implement the logic to introduce temporal error in the message.
        # Note: Skip this for now since the required image message is not provided in the message.
        return message
        
    @sub_attack
    def SL_misalignment(self, message):
        print(f"Introducing sensor-to-language misalignment in ...")
        # TODO: Implement the logic to introduce sensor-to-language misalignment in the message.
        # Note: Skip this for now since the required image message is not provided in the message.
        return message

    def _extract_objects_from_description(self, object_description):
        """extract objects from the object description."""
        import re
        
        # Pattern to match: "1. **Car ahead**: description" or similar
        pattern = r'\d+\.\s*\*\*([^*]+)\*\*:\s*([^0-9]+?)(?=\d+\.|$)'
        matches = re.findall(pattern, object_description, re.MULTILINE | re.DOTALL)
        
        objects = []
        # extract objects mentioned in double asterisks
        pattern = r'\*\*([^*]+)\*\*'
        matches = re.findall(pattern, object_description)
        
        for match in matches:
            label = match.strip().split()[0]
            objects.append({
                "label": label
            })
        return objects

    def _select_objects_for_mislabelling(self, extracted_objects):
        """Randomly select objects to mislabel."""
        if not extracted_objects:
            return []
        
        # Select 1-3 objects randomly (at least 1 if available)
        num_to_select = min(random.randint(1, 3), len(extracted_objects))
        selected = random.sample(extracted_objects, num_to_select)
        
        # Generate mislabellings for each selected object
        for obj in selected:
            obj["mislabellings"] = self._generate_mislabellings(obj["label"])
        
        return selected

    def _generate_mislabellings(self, original_label):
        """Generate mislabellings for an object label."""
        label = original_label.lower().strip()
        
        # Predefined dangerous mislabellings
        mislabelling_map = {
            "car": ["pedestrian", "bicycle", "motorcycle", "truck"],
            "truck": ["car", "bus", "emergency vehicle"],
            "pedestrian": ["car", "bicycle", "animal", "statue"],
            "bicycle": ["pedestrian", "motorcycle", "scooter"],
            "motorcycle": ["bicycle", "car", "animal"],
            "bus": ["truck", "car", "train"],
            "stop sign": ["yield sign", "speed limit sign", "traffic light"],
            "traffic light": ["stop sign", "yield sign", "pedestrian crossing"],
            "pedestrian crossing": ["traffic light", "stop sign", "bike lane"],
            "animal": ["pedestrian", "bicycle", "debris"],
            "debris": ["animal", "pedestrian", "pothole"],
            "pothole": ["debris", "animal", "shadow"]
        }
        
        # Get mislabellings or use generic ones
        suggestions = mislabelling_map.get(label, ["obstacle", "hazard", "unknown object"])
        return random.sample(suggestions, min(3, len(suggestions)))

    def _apply_mislabellings(self, original_description, objects_to_mislabel):
        """Apply the mislabellings to the original description."""
        modified_description = original_description
        
        for obj in objects_to_mislabel:
            original_label = obj["label"]
            mislabellings = obj.get("mislabellings", [])
            
            if mislabellings:
                # Choose one mislabelling randomly
                new_label = random.choice(mislabellings)
                
                # Replace in the description (case-insensitive)
                import re
                pattern = re.compile(re.escape(original_label), re.IGNORECASE)
                modified_description = pattern.sub(new_label, modified_description)
                
                print(f"Mislabelled '{original_label}' -> '{new_label}'")
        
        return modified_description
   
    def _remove_objects_from_description(self, original_description, objects_to_omit):
        """Remove selected objects from the object description."""
        import re
        
        modified_description = original_description
        
        # Create a list of labels to remove (both original labels and main nouns)
        labels_to_remove = set()
        for obj in objects_to_omit:
            labels_to_remove.add(obj['label'])
            if 'original_label' in obj:
                labels_to_remove.add(obj['original_label'])
        
        # Split description into numbered items
        items = re.split(r'(\d+\.\s*)', original_description)
        
        # Filter out items that contain objects to omit
        filtered_items = []
        item_counter = 1
        
        i = 0
        while i < len(items):
            if i + 1 < len(items) and re.match(r'\d+\.\s*', items[i]):
                # This is a numbered item
                number_part = items[i]
                content_part = items[i + 1] if i + 1 < len(items) else ""
                
                # Check if this item contains any object to omit
                should_omit = False
                for label in labels_to_remove:
                    if label.lower() in content_part.lower():
                        should_omit = True
                        break
                
                if not should_omit:
                    # Keep this item but renumber it
                    filtered_items.append(f"{item_counter}. ")
                    filtered_items.append(content_part)
                    item_counter += 1
                
                i += 2  # Skip both number and content parts
            else:
                # This is not a numbered item, keep as is
                if items[i].strip():  # Only keep non-empty parts
                    filtered_items.append(items[i])
                i += 1
        
        # Join the filtered items
        modified_description = ''.join(filtered_items).strip()
        
        # Clean up extra whitespace and newlines
        modified_description = re.sub(r'\n\s*\n', '\n\n', modified_description)
        modified_description = re.sub(r'\n+$', '', modified_description)
        
        return modified_description
   
   
   
   
   