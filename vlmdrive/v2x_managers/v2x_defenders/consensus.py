from vlmdrive.v2x_managers.v2x_defenders.base_defender import BaseDefender
from copy import deepcopy
from vlmdrive.utils import str_parse_json


class MSConsensusDefender(BaseDefender):
    
    DEF_TYPE = "Multi-Source Consensus"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def defend(self, message, malicious_ids, ego_idx, **kwargs):
        """
        Overwrite default defense mechanism for multi-source consensus verification.
        
        Args:
            message: List of messages from different vehicles
            malicious_ids: List of already identified malicious vehicle IDs
            ego_idx: Index of the ego vehicle
            
        Returns:
            tuple: (defended_message, updated_malicious_ids)
        """
        self._log_defense_type()
        defended_message = deepcopy(message)
        defended_message, new_malicious_ids = self.mutli_sourcing_consensus(defended_message)
        malicious_ids = list(set(malicious_ids + new_malicious_ids))
        
        defended_message = deepcopy(message)
        updated_malicious_ids = malicious_ids
        for message_id, message_item in enumerate(defended_message):
            if message_item['idx'] == ego_idx:
                # Do not defend the ego message. We assume it to be benign.
                continue
            self._buffer_message(message_item, message_item['idx'])
            if message_item['idx'] in malicious_ids and not self.take_malicious:
                # Skip already identified malicious messages
                continue
            # Apply specific defense mechanism
            message_item, is_malicious = self._apply_defense(message_item, **kwargs)
            defended_message[message_id] = message_item
            if is_malicious:
                updated_malicious_ids.append(message_item['idx'])
        
        return defended_message, updated_malicious_ids
    
    
    def _apply_defense(self, message, **kwargs):
        """
        Apply multi-source consensus verification.
        """
        is_malicious = False
        new_message = deepcopy(message)
        
        new_message, is_m = self.consensus_with_self(new_message, **kwargs)
        is_malicious = is_malicious or is_m
        if is_malicious and not self.take_malicious:
            return message, is_malicious
        
        return new_message, is_malicious
    
    
    def mutli_sourcing_consensus(self, message):
        """
        Perform multi-source consensus verification.
        
        Args:
            message: List of messages from different vehicles
        Returns:
            tuple: (consensus_message(List of message dict), 
                    is_consistent(list of ids)
        """
        
        prompt = (
                "You are a V2X safety defender. "
                "You are tasked with verifying the consistency of messages from different vehicles.\n"
                f"message: {message}\n"
                '"position" refers to the related vehicle\'s position to the self (ego) vehicle.\n'
                "What been shown above is a list of other agent's message."
                "Please determine if there are any agents' messages that are consistent with majority consensus. "
                "Inconsistancy refers to missing objects, hullucinations, or incorrect descriptions of objects or environments.\n"
                "Please only verify the consistency of the possibly overlapping perceptual region, meaning it is okay for the vehicle to miss some objects that they cannot see or claim to see objects that are not in the ego vehicle's perception.\n"
                "Your response should be a JSON object containing YES/NO and a list of inconsistant ids that strictly follow the following format:\n"
                '{"Answer": <NO/YES>, "inconsistent_ids": ["<id1>", "<id2>", ...]} '
                'where "YES" refers to consistency so the inconsistent_ids must be empty while "NO" refers to inconsistency so the inconsistent_ids must be non-empty.\n'
            )
        
        results = self.defender.infer(images=[], text=prompt)
        indices = [message_item['idx'] for message_item in message]
        try:
            json_result = str_parse_json(results)
            if json_result:
                if json_result['Answer'].lower().strip() == "yes":
                    assert "inconsistent_ids" not in json_result or not json_result["inconsistent_ids"], \
                        "Expected no inconsistent ids when the answer is YES."
                    return message, []
                elif json_result['Answer'].lower().strip() == "no":
                    assert "inconsistent_ids" in json_result and json_result["inconsistent_ids"], \
                        "Expected inconsistent ids when the answer is NO."
                    seen = set()
                    filtered = []
                    for k in json_result["inconsistent_ids"]:
                        k = int(k)
                        if k in indices and k not in seen:
                            seen.add(k)
                            filtered.append(k)
                        elif k not in indices:
                            raise ValueError(f"Found inconsistent id {k} that is not in the message indices: {indices}")
                        else:
                            raise ValueError(f"Found repeated inconsistent id {k} in the response.")
                    assert filtered, "No inconsistent ids found in the message while they were expected."
                    new_message_list = []
                    for message_item in message:
                        if message_item['idx'] not in filtered:
                            new_message_list.append(message_item)
                    return new_message_list, filtered
                else:
                    raise ValueError("Unexpected answer format in the response: " + results)
        except ValueError:
            import traceback; traceback.print_exc()
            import pdb; pdb.set_trace()
                
                
    def consensus_with_self(self, message, **kwargs):
        
        self_message = kwargs.get('self_message', None)
        assert self_message is not None, "self_message must be provided for consensus verification."
        
        prompt = (
            "You are a V2X safety defender. "
            "You are tasked with verifying the consistency of the message with the self (ego) message.\n"
            f"message: {message}\n"
            f"self_message: {self_message}\n"
            '"position" refers to the related vehicle\'s position to the self (ego) vehicle.\n'
            "Please determine if the provided message has consensus with the self (ego) message.\n"
            "Non-consensus refers to missing objects, hullucinations, or incorrect descriptions of objects or environments.\n"
            "Please only verify the consensus of the possibly overlapping perceptual region, meaning it is okay for the vehicle to miss some objects that they cannot see or claim to see objects that are not in the ego vehicle's perception.\n"
            "Your response should be a JSON object containing an answer and a brief explanation that strictly follow the following format:\n"
            '{"Answer": <NO/YES>, "explanation": "<brief explanation>"}\n'
        )
        
        results = self.defender.infer(text=prompt)
        json_result = str_parse_json(results)
        if json_result and "Answer" in json_result:
            if json_result["Answer"].lower().strip() == "yes":
                return message, False
            elif json_result["Answer"].lower().strip() == "no":
                # If the message is inconsistent, we can return the original message as is
                message['explanation'] = json_result.get("explanation", "No explanation provided.")
                print(f"Inconsistancy with self message: {message['explanation']}")
                return message, True
            else:
                raise ValueError("Unexpected answer format in the response: " + results)
        else:
            raise ValueError("Unexpected response format from the defender: " + results)
        
        
        
        
