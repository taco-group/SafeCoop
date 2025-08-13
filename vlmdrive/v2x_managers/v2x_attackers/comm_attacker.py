from vlmdrive.v2x_managers.v2x_attackers.base_attacker import BaseAttacker, sub_attack
import random
import time


class CommAttacker(BaseAttacker):
    
    ATT_TYPE = "Communication Attack"
    
    def __init__(self):
        super().__init__()

    @sub_attack
    def message_tampering(self, message):
        print(f"Tampering with message integrity ...")
        # TODO: Implement logic to tamper with message integrity or replay attacks
        # Could modify any field in the message to simulate corruption
        return message

    @sub_attack
    def clock_synchronization_error(self, message):
        print(f"Introducing clock synchronization error ...")
        # TODO: Implement logic to introduce timing errors
        # Could add timestamp fields and manipulate them
        return message
    
    @sub_attack
    def identity_spoofing(self, message):
        print(f"Spoofing vehicle identity ...")
        # TODO: Implement logic to spoof vehicle identity (Sybil attack)
        # Could modify the 'idx' field to impersonate other vehicles
        return message

    @sub_attack
    def byzantine_collusion(self, message):
        print(f"Executing Byzantine collusion attack ...")
        # TODO: Implement logic for coordinated malicious behavior
        # Could coordinate multiple vehicles to send conflicting information
        # Leave it empty for now as it may need extra information/interface.
        return message
        
    @sub_attack
    def denial_of_service(self, message):
        print(f"Executing denial of service attack ...")
        # TODO: Implement logic to flood or block communication
        # Could duplicate messages or inject noise
        return message