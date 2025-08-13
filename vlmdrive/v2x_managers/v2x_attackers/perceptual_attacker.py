from vlmdrive.v2x_managers.v2x_attackers.base_attacker import BaseAttacker, sub_attack
import random


class PerceptualAttacker(BaseAttacker):
    
    ATT_TYPE = "Perceptual Attack"
    
    def __init__(self):
        super().__init__()

    @sub_attack
    def omission(self, message):
        print(f"Omitting info ...")
        # TODO: Implement the logic to omit information from the message.

    @sub_attack
    def hallucination(self, message):
        print(f"Adding hallucination ...")
        # TODO: Implement the logic to add hallucination to the message.
    
    @sub_attack
    def mislabeling(self, message):
        print(f"Mislabeling info ...")
        # TODO: Implement the logic to mislabel information in the message.

    @sub_attack
    def localization_error(self, message):
        print(f"Introducing localization error ...")
        # TODO: Implement the logic to introduce localization error in the message.
        
    @sub_attack
    def temporal_error(self, message):
        print(f"Introducing temporal error in ...")
        # TODO: Implement the logic to introduce temporal error in the message.
        
    @sub_attack
    def SL_misalignment(self, message):
        print(f"Introducing sensor-to-language misalignment in ...")
        # TODO: Implement the logic to introduce sensor-to-language misalignment in the message.
        # Note: Skip this for now since the required image message is not provided in the message.
   
   
   
   
   