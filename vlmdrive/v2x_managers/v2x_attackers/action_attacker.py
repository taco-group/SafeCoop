from vlmdrive.v2x_managers.v2x_attackers.base_attacker import BaseAttacker, sub_attack
import random


class ActionAttacker(BaseAttacker):
    
    ATT_TYPE = "Action Attack"
    
    def __init__(self):
        super().__init__()

    @sub_attack
    def override_safety(self, message):
        print(f"Overriding safety constraints ...")
        # TODO: Implement logic to override safety constraints in action planning
        # Could modify intent_description to suggest unsafe actions
        return message

    @sub_attack
    def ambiguous_commands(self, message):
        print(f"Injecting ambiguous commands ...")
        # TODO: Implement logic to make commands ambiguous or under-specified
        # Could modify intent_description to be vague or contradictory
        return message
    
    @sub_attack
    def adversarial_context(self, message):
        print(f"Injecting adversarial context ...")
        # TODO: Implement logic to inject false context that leads to unsafe actions
        # Could modify scene_description to fabricate emergency situations
        return message

    @sub_attack
    def ethical_manipulation(self, message):
        print(f"Applying ethical manipulation ...")
        # TODO: Implement logic to manipulate ethical decision making
        # Could modify object_description to bias decisions against certain entities
        return message