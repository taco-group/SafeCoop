from vlmdrive.v2x_managers.v2x_attackers.perceptual_attacker import PerceptualAttacker
from vlmdrive.v2x_managers.v2x_attackers.action_attacker import ActionAttacker
from vlmdrive.v2x_managers.v2x_attackers.comm_attacker import CommAttacker

from vlmdrive.v2x_managers.v2x_defenders.firewall import FirewallDefender
from vlmdrive.v2x_managers.v2x_defenders.consistency import LPConsistencyDefender
from vlmdrive.v2x_managers.v2x_defenders.consensus import MSConsensusDefender

from vlmdrive.vlm.vlm_planner_utils import configure_vlm_helpers

import math




class V2XManager():
    
    def __init__(self, 
                 atker_config, 
                 defender_config,
                 self_id, 
                 atker_ids, 
                 ego_num):
        
        self.self_id = self_id
        self.atker_ids = atker_ids
        self.ego_num = ego_num
        
        # Initialize attacker and defender modules
        self._init_atker_defender(atker_config=atker_config, defender_config=defender_config)
        
        self.pred_malicious_ids = [] # List to store predicted malicious vehicle IDs for evaluation
        
    def _init_atker_defender(self, atker_config, defender_config):
        """
        Initialize the attacker and defender configurations.
        """
        if atker_config is None:
            raise ValueError("Attacker configuration cannot be None.")
        if defender_config is None:
            raise ValueError("Defender configuration cannot be None.")
                
        atker = configure_vlm_helpers(
            name=atker_config["name"],
            api_model_name=atker_config["api_model_name"],
            api_base_url=atker_config["api_base_url"],
            api_key=atker_config["api_key"],
            image_placeholder=atker_config["IMAGE_PLACEHOLDER"],
        )['atker']
        
        defender = configure_vlm_helpers(
            name=defender_config["name"],
            api_model_name=defender_config["api_model_name"],
            api_base_url=defender_config["api_base_url"],
            api_key=defender_config["api_key"],
            image_placeholder=defender_config["IMAGE_PLACEHOLDER"],
        )['defender']
        
        self._initialize_attackers(atker)
        self._initialize_defenders(defender, take_malicious=defender_config['take_malicious'])
        
    def _initialize_attackers(self, atker):
        self.perceptual_attacker = PerceptualAttacker(atker=atker)
        self.action_attacker = ActionAttacker(atker=atker)
        self.comm_attacker = CommAttacker(atker=atker)
        
    def _initialize_defenders(self, defender, **kwargs):
        self.firewall_defender = FirewallDefender(defender=defender, **kwargs)
        self.lpc_defender = LPConsistencyDefender(defender=defender, **kwargs)
        self.msc_defender = MSConsensusDefender(defender=defender, **kwargs)
    
    def simulate_attack(self, message, ego_idx):
        """
        Simulate an attack using the attacker module.
        """
        if ego_idx != self.self_id:
            # If the ego vehicle is not the self vehicle, we assume it is benign.
            return message
        
        message = self.perceptual_attacker.attack(message, ego_idx)
        message = self.action_attacker.attack(message, ego_idx)
        message = self.comm_attacker.attack(message, ego_idx)
        return message
        
    def simulate_defense(self, message, ego_idx):
        """
        Simulate a defense using the defender module.
        """
        malicious_ids = []
        if ego_idx != self.self_id:
            # If the ego vehicle is not the self vehicle, we assume it is benign, no need to defend.
            return message, malicious_ids
        
        message, malicious_ids = self.firewall_defender.defend(message, malicious_ids, ego_idx)
        message, malicious_ids = self.lpc_defender.defend(message, malicious_ids, ego_idx)
        message, malicious_ids = self.msc_defender.defend(message, malicious_ids, ego_idx)
        
        # Update the predicted malicious IDs for evaluation
        self.pred_malicious_ids.append(malicious_ids)
        
        return message, malicious_ids
    
    
    def clean_up(self):
        """
        Clean up predicted malicious IDs
        """
        self.pred_malicious_ids = []
    
    def evaluate(self, gamma=0.95, lam=1.0, eps=1e-9):
        """
        Please refer to doc/eval_metric.md for detailed description of the evaluation metrics.
        """
        
        atk_idx = self.atker_ids
        pred = self.pred_malicious_ids
        N = self.ego_num
        
        # atk_idx: set or list of attacker indices (size k)
        # pred: list length T; each element is an iterable of predicted attacker indices at time t
        A = set(atk_idx); k = len(A); T = len(pred)
        F1s, Jaccs, ws = [], [], []
        # per-agent sequences
        hat = {i: [0]*T for i in range(N)}
        for t, P in enumerate(pred, start=1):
            P = set(P)
            TP = len(P & A); FP = len(P - A); FN = len(A - P)
            prec = TP / (TP + FP + eps)
            rec  = TP / (TP + FN + eps)
            f1 = 2*prec*rec / (prec + rec + eps)
            jacc = TP / (len(P | A) + eps)
            F1s.append(f1); Jaccs.append(jacc); ws.append(gamma**(t-1))
            for i in P: hat[i][t-1] = 1

        F1_mean = sum(F1s)/T
        J_mean  = sum(Jaccs)/T
        WF1     = sum(w*f for w,f in zip(ws,F1s)) / (sum(ws)+eps)
        WJacc   = sum(w*j for w,j in zip(ws,Jaccs)) / (sum(ws)+eps)

        # LADS

        tau = {}
        for i in A:
            try:
                t_first = hat[i].index(1) + 1
                tau[i] = t_first
            except ValueError:
                tau[i] = math.inf
        c = [(1 - (t-1)/T) if t != math.inf else 0.0 for t in tau.values()]
        # false positive rate over time for normal agents
        normals = [j for j in range(N) if j not in A]
        b = [sum(hat[j])/T for j in normals] if normals else [0.0]
        LADS = (sum(c)/max(k,1)) - lam * (sum(b)/max(len(normals),1))

        # stability
        flips = 0
        for i in range(N):
            seq = hat[i]
            flips += sum(int(seq[t]!=seq[t-1]) for t in range(1,T))
        FlipRate = flips / (N*max(T-1,1))

        return {
            "F1_mean": F1_mean, "Jacc_mean": J_mean,
            "WF1": WF1, "WJacc": WJacc,
            "LADS": LADS, "FlipRate": FlipRate,
            "tau_stats": {
                "median": float("inf") if not c or all(v==0 for v in c) else sorted([t for t in tau.values() if t!=math.inf])[len([t for t in tau.values() if t!=math.inf])//2] if any(t!=math.inf for t in tau.values()) else float("inf"),
                "miss_rate": sum(1 for t in tau.values() if t==math.inf) / max(k,1)
            }
        }
        
        

if __name__ == "__main__":
    
    from pprint import pprint
    
    class V2XManagerTest(V2XManager):
        """
        Reimplementation of V2XManager for testing purposes. (Disable initialization of attackers and defenders)
        """
        def __init__(self, atker_ids, ego_num, pred):
            
            self.atker_ids = atker_ids
            self.ego_num = ego_num
            
            self.pred_malicious_ids = pred # List to store predicted malicious vehicle IDs for evaluation
    
    ############################
    # Testing evaluate metrics #
    ############################
    
    # Perfect prediction example
    atk_idx = [1, 3]
    pred = [
        [1, 3],     # t=1 (perfect)
        [1, 3],     # t=2 (perfect)
        [1, 3],     # t=3 (perfect)
    ]
    N = 5
    
    v2x_manager = V2XManagerTest(atker_ids=atk_idx, ego_num=N, pred=pred)
    results = v2x_manager.evaluate()
    pprint("Perfect Prediction Results:")
    pprint(results)
    print()
    
    
    
    # Delayed detection + some false positives
    atk_idx = [2, 5]
    pred = [
        [],          # t=1: predict nothing (miss both attackers)
        [2],         # t=2: detect agent 2 only
        [2, 5, 4],   # t=3: detect both but with a false positive (4)
        [2, 5],      # t=4: correct
    ]
    N = 6
    
    v2x_manager = V2XManagerTest(atker_ids=atk_idx, ego_num=N, pred=pred)
    results = v2x_manager.evaluate()
    pprint("Delayed Detection + False Positives Results:")
    pprint(results)
    print()
    
    # Noisy predictions (unstable)
    atk_idx = [0]
    pred = [
        [0],     # t=1: correct
        [],      # t=2: miss
        [0, 2],  # t=3: detect + false positive
        [0],     # t=4: back to correct
        [1],     # t=5: completely wrong
    ]
    N = 4
    
    v2x_manager = V2XManagerTest(atker_ids=atk_idx, ego_num=N, pred=pred)
    results = v2x_manager.evaluate()
    pprint("Noisy Predictions Results:")
    pprint(results)
    print()
    
    # Large imbalance
    atk_idx = [7]
    pred = [
        [],            # t=1: miss
        [2, 3],        # t=2: wrong predictions
        [7],           # t=3: correct
        [7, 9],        # t=4: correct + false positive
        [7],           # t=5: correct
    ]
    N = 10
    
    v2x_manager = V2XManagerTest(atker_ids=atk_idx, ego_num=N, pred=pred)
    results = v2x_manager.evaluate()
    pprint("Large Imbalance Results:")
    pprint(results)
    
    
    # No predictions (all benign)
    atk_idx = [7]
    pred = [
        [],            # t=1: miss
        [],        # t=2: wrong predictions
        [],           # t=3: correct
        [],        # t=4: correct + false positive
        [],           # t=5: correct
    ]
    N = 10
    
    v2x_manager = V2XManagerTest(atker_ids=atk_idx, ego_num=N, pred=pred)
    results = v2x_manager.evaluate()
    pprint("No Predictions Results:")
    pprint(results)
        
    
    

   
    
'''
An message exmample for your reference:

[{'ego_yaw': 1.5719406604766846,
  'idx': 1,
  'intent_description': '  Target is front-left (~34°). There is a cyclist '
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
                        'after reaching it.\n'}]
'''



