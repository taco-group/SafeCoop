from dataclasses import dataclass

@dataclass
class AttackCounter:
    threshold: int = 150  # show overlay after 150 ticks
    count: int = 0
    message: str = ""
    defense_message: str = ""

    def tick(self, n: int = 1) -> None:
        self.count += n

    def start_attack(self) -> bool:
        """True once we've reached the threshold (keeps returning True afterwards)."""
        return self.count >= self.threshold

    def pop_ready(self) -> bool:
        """Edge-trigger: returns True once, then resets the counter."""
        if self.count >= self.threshold:
            self.count = 0
            return True
        return False

    def reset(self) -> None:
        self.count = 0

    def set_message(self, msg: str):
        pass
        # if self.message:
        #     self.message += f"\n{msg}"
        # else:
        #     self.message = msg
        
    def set_defense_message(self):
        pass
        # self.defense_message = f"Defense Activated"
        
    def append_message(self, msg: str):
        pass
        # if self.message:
        #     self.message += f"\n{msg}"
        # else:
        #     self.message = msg
            
    def reset_messages(self) -> None:
        self.message = ""
        self.defense_message = ""

# module-level instance shared across files
attack_counter = AttackCounter(threshold=0) # threshold is when to start the attack 