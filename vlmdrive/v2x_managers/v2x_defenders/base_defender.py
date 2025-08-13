from abc import ABC, abstractmethod
from copy import deepcopy


class BaseDefender(ABC):
    
    def __init__(self):
        pass
    
    def _log_defense_type(self):
        """
        Log the defense type.
        """
        print(f"Defense Type: {self.DEF_TYPE}")
        
    def defend(self, message, malicious_ids, ego_idx):
        """
        Apply defense mechanism to the message and identify malicious vehicles.
        
        Args:
            message: List of messages from different vehicles
            malicious_ids: List of already identified malicious vehicle IDs
            ego_idx: Index of the ego vehicle
            
        Returns:
            tuple: (defended_message, updated_malicious_ids)
        """
        self._log_defense_type()
        defended_message = deepcopy(message)
        for message_id, message_item in enumerate(defended_message):
            if message_item['idx'] == ego_idx:
                # Do not defend the ego message. We assume it to be benign.
                continue
            if message_id in malicious_ids:
                # Skip already identified malicious messages
                continue
            # Apply specific defense mechanism
            message_item, is_malicious = self._apply_defense(message_item)
            defended_message[message_id] = message_item
            if is_malicious:
                updated_malicious_ids.append(message_item['idx'])
        
        return defended_message, updated_malicious_ids
    
    @abstractmethod
    def _apply_defense(self, message, ego_idx):
        """
        Apply the specific defense mechanism.
        
        Args:
            message: List of messages from different vehicles
            ego_idx: Index of the ego vehicle
            
        Returns:
            tuple: (defended_message, new_malicious_ids)
        """
        pass