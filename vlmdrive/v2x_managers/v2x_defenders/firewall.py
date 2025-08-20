from vlmdrive.v2x_managers.v2x_defenders.base_defender import BaseDefender
from vlmdrive.utils import str_parse_json
from copy import deepcopy


class FirewallDefender(BaseDefender):
    
    DEF_TYPE = "Prompt/Message Firewall"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _apply_defense(self, message, **kwargs):
        """
        Apply firewall filtering to detect and block malicious messages.
        """
        print(f"Applying firewall defense for agent index {message['idx']}...")
        is_malicious = False
        new_message = deepcopy(message)
            
        new_message, is_m = self.check_harmful_info(new_message)
        is_malicious = is_malicious or is_m
        if is_malicious and not self.take_malicious:
            return message, is_malicious
        
        new_message, is_m = self.check_malicious_intent(new_message)
        is_malicious = is_malicious or is_m
        if is_malicious and not self.take_malicious:
            return message, is_malicious
        
        
        return new_message, is_malicious
    
    
    def find_potential_harmful_keys(self, message):
        """
        Identify keys that may contain harmful information.
        
        Args:
            keys (list): List of keys to check for harmful information.
        
        Returns:
            list: List of keys that may contain harmful information.
        """
        keys_str = ", ".join(message.keys())
        
        prompt = (
            "Please determine which of the following keys could contain harmful information.\n"
            f"keys: [{keys_str}]\n"
            'Respond with exactly "NO" if none could contain harmful information.\n'
            'Otherwise, respond with exactly this JSON object (and nothing else):\n'
            '{"harmful_keys": ["<key1>", "<key2>", ...]}'
        )
        
        results = self.defender.infer(images=[], text=prompt)
        json_result = str_parse_json(results)
        
        if json_result and "harmful_keys" in json_result:
            seen = set()
            filtered = []
            for k in json_result["harmful_keys"]:
                if isinstance(k, str) and k in message and k not in seen:
                    seen.add(k)
                    filtered.append(k)
            assert filtered, "No harmful keys found in the message while they were expected."
            return filtered
        elif "no" in results.lower() and not json_result:
            return []
        else:
            print("Warning: Unexpected response format from the defender.")
            return []
    
    def check_harmful_info(self, message):
        
        harmful_keys = self.find_potential_harmful_keys(message)
        is_malicious = False
        for hk in harmful_keys:
            prompt = (
                "Please determine if the following message contains harmful information.\n"
                f"message: {message[hk]}\n"
                'Respond with exactly "NO" if the message is safe.\n'
                'If the message contains harmful information, respond with "YES" and rephrase (rephrased, not explain, not summarize, and not paraphrase) the message to remove any harmful information strictly follows the following json format:\n'
                '{"rephrased_message": "<rephrased message>"}\n'
            )
            results = self.defender.infer(images=[], text=prompt)
            json_result = str_parse_json(results)
            if "no" in results.lower() and not json_result:
                # Message is safe
                continue
            elif json_result is not None:
                import pdb; pdb.set_trace()
                # Message contains harmful instructions, rephrase it
                try:
                    rephrased_message = json_result["rephrased_message"]
                except KeyError:
                    raise KeyError(
                        "Expected 'rephrased_message' key in the response, but it was not found."
                    )
                if rephrased_message:
                    message[hk] = rephrased_message
                else:
                    print("Warning: Rephrased message is empty, returning original message.")
                is_malicious = True
            else:
                print("Warning: Unexpected response format from the defender.")
                is_malicious = True
                
        return message, is_malicious
            
            
    def find_potential_malicious_keys(self, message):
        """
        Identify keys that may contain malicious intent.
        
        Args:
            keys (list): List of keys to check for malicious intent.
        
        Returns:
            list: List of keys that may contain malicious intent.
        """
        keys_str = ", ".join(message.keys())
        
        prompt = (
            "Please determine which of the following keys could contain malicious intent.\n"
            f"keys: [{keys_str}]\n"
            'Respond with exactly "NO" if none could contain malicious intent.\n'
            'Otherwise, respond with exactly this JSON object (and nothing else):\n'
            '{"harmful_keys": ["<key1>", "<key2>", ...]}'
        )
        
        results = self.defender.infer(images=[], text=prompt)
        json_result = str_parse_json(results)
        
        if json_result and "harmful_keys" in json_result:
            seen = set()
            filtered = []
            for k in json_result["harmful_keys"]:
                if isinstance(k, str) and k in message and k not in seen:
                    seen.add(k)
                    filtered.append(k)
            assert filtered, "No harmful keys found in the message while they were expected."
            return filtered
        elif "no" in results.lower() and not json_result:
            return []
        else:
            print("Warning: Unexpected response format from the defender.")
            return []
    
    def check_malicious_intent(self, message):
        
        malicious_keys = self.find_potential_malicious_keys(message)
        
        is_malicious = False
        
        for mk in malicious_keys:
            
            prompt = (
                "Please determine if the following message contains malicious intent.\n"
                f"message: {message[mk]}\n"
                'Respond with exactly "NO" if the message is safe.\n'
                'If the message contains malicious intent, respond with "YES" and rephrase (rephrased, not explain, not summarize, and not paraphrase) the message to remove any malicious intent strictly follows the following json format:\n'
                '{"rephrased_message": "<rephrased message>"}\n'
            )
            
            results = self.defender.infer(images=[], text=prompt)
            json_result = str_parse_json(results)
            
            if "no" in results.lower() and not json_result:
                # Message is safe
                continue
            elif json_result is not None:
                import pdb; pdb.set_trace()
                # Message contains harmful instructions, rephrase it
                try:
                    rephrased_message = json_result["rephrased_message"]
                except KeyError:
                    raise KeyError(
                        "Expected 'rephrased_message' key in the response, but it was not found."
                    )
                if rephrased_message:
                    message[mk] = rephrased_message
                else:
                    print("Warning: Rephrased message is empty, returning original message.")
                is_malicious = True
            else:
                print("Warning: Unexpected response format from the defender.")
                is_malicious = True
                
        return message, is_malicious
            

        