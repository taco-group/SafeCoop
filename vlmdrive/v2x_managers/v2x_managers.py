from vlmdrive.v2x_managers.v2x_attackers.perceptual_attacker import PerceptualAttacker
from vlmdrive.v2x_managers.v2x_attackers.action_attacker import ActionAttacker
from vlmdrive.v2x_managers.v2x_attackers.comm_attacker import CommAttacker

from vlmdrive.v2x_managers.v2x_defenders.firewall import FirewallDefender
from vlmdrive.v2x_managers.v2x_defenders.consistency import LPConsistencyDefender
from vlmdrive.v2x_managers.v2x_defenders.consensus import MSConsensusDefender



class V2XManager():
    
    def __init__(self):
        
        self.perceptual_attacker = PerceptualAttacker()
        self.action_attacker = ActionAttacker()
        self.comm_attacker = CommAttacker()
        
        self.firewall_defender = FirewallDefender()
        self.lpc_defender = LPConsistencyDefender()
        self.msc_defender = MSConsensusDefender()
    
    def simulate_attack(self, message, ego_idx):
        """
        Simulate an attack using the attacker module.
        """
        message = self.perceptual_attacker.attack(message, ego_idx)
        message = self.action_attacker.attack(message, ego_idx)
        message = self.comm_attacker.attack(message, ego_idx)
        return message
        
    def simulate_defense(self, message, ego_idx):
        """
        Simulate a defense using the defender module.
        """
        malicious_ids = []
        message, malicious_ids = self.firewall_defender.defend(message, malicious_ids, ego_idx)
        message, malicious_ids = self.lpc_defender.defend(message, malicious_ids, ego_idx)
        message, malicious_ids = self.msc_defender.defend(message, malicious_ids, ego_idx)
        
        return message, malicious_ids
    
    

   
    
'''
An message exmample for your reference:

{'ego_yaw': 1.5719406604766846,
  'idx': 1,
  'intent_description': 'Target is front-left (~34°). There is a cyclist '
                        'crossing left ahead — yield first.\n'
                        '\n'
                        'Plan (concise):\n'
                        '- Steer slightly-left toward bearing ≈34° to aim at '
                        'the target.\n'
                        '- Gently decelerate to a safe approach speed (reduce '
                        'speed briefly to ~3–6 m/s) to yield to the cyclist '
                        'and ensure no collision.\n'
                        '- When the path is clear, complete the leftward '
                        'steering to reach the target, then straighten wheels '
                        'and continue forward.\n'
                        '- After passing the target resume acceleration up to '
                        'the speed limit (≤20 m/s).\n'
                        '\n'
                        'Maintain lookout and abort/stop if the cyclist or any '
                        'vehicle encroaches into our lane.',
  'object_description': '- Cyclist — left/center of intersection, riding '
                        'across the carriageway from left to right; actively '
                        'crossing the intersection and will continue into your '
                        'path.\n'
                        '- Parked/standing vehicle — right curb at corner '
                        '(white car), stationary near the crosswalk; stopped '
                        'or preparing to re-enter traffic, may pull out or '
                        'block sightlines.\n'
                        '- Distant vehicles/pedestrians — far center/right '
                        'beyond intersection, moving slowly or stationary at '
                        'the lights; potential to enter the intersection when '
                        'signals change.',
  'position': array([-1.30000014e+01,  5.96360337e-06]),
  'scene_description': '- Clear, sunny daytime with strong sunlight and good '
                       'visibility; dry pavement.\n'
                       '- Approaching a multi-lane signalized intersection '
                       'with overhead traffic lights.\n'
                       '- Light traffic overall; a cyclist is crossing '
                       'left-to-right through the intersection (potential '
                       'hazard).\n'
                       '- Several parked/stopped cars on the right near the '
                       'curb limiting lateral clearance.\n'
                       '- Well-marked lanes and crosswalks; sidewalks and '
                       'street trees line both sides.',
  'target_description': 'The target is 3.50012 meters to your left and 5.13972 '
                        'meters to your front. \n'
                        'The target is not an endpoint—continue moving forward '
                        'after reaching it.\n'}
'''