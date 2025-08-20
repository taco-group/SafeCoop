from vlmdrive.v2x_managers.v2x_defenders.base_defender import BaseDefender
from copy import deepcopy
from vlmdrive.utils import str_parse_json


class LPConsistencyDefender(BaseDefender):
    
    DEF_TYPE = "Language-Perception Consistency Verification"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _apply_defense(self, message, **kwargs):
        """
        Apply language-perception consistency verification.
        
        There are two types of language-perception consistency checks:
        1. actors_lpc: If the message contains perceptual data, we check if the language description
        matches the perceptual data.
        2. ego_lpc: We use the ego vehicle's perception to verify the consistency of other vehicles'
        language descriptions. By doing so, relative positions and rotations need to be
        considered.
        
        """
        print(f"Applying language-perception consistency verification for agent index {message['idx']}...")
        is_malicious = False
        
        new_message = deepcopy(message)
            
        new_message, is_m = self.actors_lpc(new_message)
        is_malicious = is_malicious or is_m
        if is_malicious and not self.take_malicious:
            return message, is_malicious
        
        
        new_message, is_m = self.ego_lpc(new_message, front_image_ego=kwargs.get('front_image_ego', None))
        is_malicious = is_malicious or is_m
        if is_malicious and not self.take_malicious:
            return message, is_malicious
        
        
        return new_message, is_malicious
    
    
    def find_perceptual_related_keys(self, message):
        """
        Identify keys that may contain perceptual information.
        """
        
        keys_str = ", ".join(message.keys())
        
        prompt = (
            "Please determine which of the following keys could contain perceptual information.\n"
            f"keys: [{keys_str}]\n"
            'Respond with exactly "NO" if none could contain harmful information.\n'
            'Otherwise, respond with exactly this JSON object (and nothing else):\n'
            '{"perceptual_related_keys": ["<key1>", "<key2>", ...]}'
        )
        
        results = self.defender.infer(images=[], text=prompt)
        json_result = str_parse_json(results)
        
        if json_result and "perceptual_related_keys" in json_result:
            seen = set()
            filtered = []
            for k in json_result["perceptual_related_keys"]:
                if isinstance(k, str) and k in message and k not in seen:
                    seen.add(k)
                    filtered.append(k)
            assert filtered, "No perceptual related keys found in the message while they were expected."
            return filtered
        elif "no" in results.lower() and not json_result:
            return []
        else:
            print("Warning: Unexpected response format from the defender.")
            return []
    
    
    def actors_lpc(self, message):
        """
        Check language-perception consistency for actors.
        
        Args:
            message: The message containing actor information.
            
        Returns:
            tuple: (message, is_malicious)
        """
        # Implement the logic to check language-perception consistency for actors
        is_malicious = False
        
        # Placeholder for actual implementation
        # TODO: Has not been implemented since perceptual data is not available at this version.
        
        return message, is_malicious
    
    
    def ego_lpc(self, message, front_image_ego=None):
        # Chack if the ego vehicle's perception matches the language descriptions of other vehicles.
        assert front_image_ego is not None, "Front image of ego vehicle is required for ego LPC."
        perceptual_related_keys = self.find_perceptual_related_keys(message)
        is_malicious = False
        
        for prk in perceptual_related_keys:
            try:
                prompt = (
                        "The following image is the front view of the ego vehicle's perception"
                        f"{self.image_placeholder}\n"
                        f"The following message is a language description of other vehicles with the relative position {message['position'].tolist()}.\n"
                        f"message: {message[prk]}\n"
                        "Please determine if the following language description of other vehicles is consistent with the ego vehicle's perception.\n"
                        "Inconsistancy refers to missing objects, hullucinations, or incorrect descriptions of objects or environments.\n"
                        "Please only verify the consistency of the possibly overlapping perceptual region, meaning it is okay for the vehicle to miss some objects that they cannot see or claim to see objects that are not in the ego vehicle's perception.\n"
                        "Your response should be a JSON object containing an answer and a with a brief explanation that strictly follow the following format:\n"
                        '{"Answer": <NO/YES>, "explanation": "<brief explanation>"}\n'
                    )
                results = self.defender.infer(images=[front_image_ego], text=prompt)
                json_result = str_parse_json(results)
                if json_result and "Answer" in json_result:
                    if json_result["Answer"].lower() == "no":
                        is_malicious = True
                        # If the message is inconsistent, there is no way to recover the correct message therefore we drop it entirely.
                        message = ""
                        # print(f"Malicious message detected: {json_result['explanation']}")
                    elif json_result["Answer"].lower() == "yes":
                        pass
                        # print(f"Message is consistent: {json_result['explanation']}")
                    else:
                        raise ValueError("Unexpected answer format in the response: " + results)
                else:
                    raise KeyError(
                        "Expected 'Answer' key in the response, but it was not found."
                    )
            except ValueError:
                import traceback; traceback.print_exc()
                import pdb; pdb.set_trace()
        return message, is_malicious
                
        
        
        
        
        
        