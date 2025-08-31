from vlmdrive.v2x_managers.v2x_attackers.perceptual_attacker import PerceptualAttacker
from vlmdrive.v2x_managers.v2x_attackers.action_attacker import ActionAttacker
from vlmdrive.v2x_managers.v2x_attackers.comm_attacker import JammingAttacker, ReplayAttacker, SybilAttacker

from vlmdrive.v2x_managers.v2x_defenders.firewall import FirewallDefender
from vlmdrive.v2x_managers.v2x_defenders.consistency import LPConsistencyDefender
from vlmdrive.v2x_managers.v2x_defenders.consensus import MSConsensusDefender

from vlmdrive.vlm.vlm_planner_utils import configure_vlm_helpers
from vlmdrive.utils import run_coro_blocking

import math
from copy import deepcopy
import asyncio

import time
from pprint import pprint

class V2XManager:
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

        # For evaluation across timesteps
        self.pred_malicious_ids = []  # List[List[int]]

        # Timing from the most recent defense run
        self.last_defense_timing = None  # dict populated by simulate_defense / _simulate_defense_async

        self.trust_score_system = defender_config.get("trust_score_system", True)
        self.trust_score_threshold = defender_config.get("trust_score_threshold", 2)


    def _init_atker_defender(self, atker_config, defender_config):
        """Initialize the attacker and defender configurations."""
        if atker_config is None:
            raise ValueError("Attacker configuration cannot be None.")
        if defender_config is None:
            raise ValueError("Defender configuration cannot be None.")

        self.atker_aysnc_mode = atker_config.get("async_mode", False)  # attacker async not implemented yet
        self.defender_async_mode = defender_config.get("async_mode", True)

        atker_helpers = configure_vlm_helpers(
            name=atker_config["name"],
            provider=atker_config["provider"],
            api_model_name=atker_config["api_model_name"],
            api_base_url=atker_config["api_base_url"],
            api_key=atker_config["api_key"],
            image_placeholder=atker_config["IMAGE_PLACEHOLDER"],  # <== use attacker's placeholder
            async_mode=self.atker_aysnc_mode,
        )
        atker = atker_helpers["atker"]

        defender_helpers = configure_vlm_helpers(
            name=defender_config["name"],
            provider=defender_config["provider"],
            api_model_name=defender_config["api_model_name"],
            api_base_url=defender_config["api_base_url"],
            api_key=defender_config["api_key"],
            image_placeholder=defender_config["IMAGE_PLACEHOLDER"],
            async_mode=self.defender_async_mode,
        )
        defender = defender_helpers["defender"]

        atk_methods = atker_config.get("attack_methods", ["jamming", "replay", "spoofing", "sybil"])
        self.with_sybil = "sybil" in atk_methods
        self.sybil_num = atker_config.get("sybil_num", 3)
        self._initialize_attackers(
            atker,
            atk_methods=atk_methods,
            message_buffer_size=atker_config.get("message_buffer_size", 20),
            IMAGE_PLACEHOLDER=atker_config["IMAGE_PLACEHOLDER"],
            ego_num=self.ego_num,
            perceptual_attacker_prompt_template=atker_config.get("perceptual_attacker_prompt_template", {}),
            sybil_num=self.sybil_num
        )
        self._initialize_defenders(
            defender,
            defense_methods=defender_config.get("defense_methods", ["firewall", "lpc", "msc"]),
            message_buffer_size=defender_config.get("message_buffer_size", 20),
            IMAGE_PLACEHOLDER=defender_config["IMAGE_PLACEHOLDER"],
            with_explanation=defender_config.get("with_explanation", False),
            trust_score_system=defender_config.get("trust_score_system", True),
        )

    def _initialize_attackers(self, atker, atk_methods, **kwargs):
        if 'jamming' in atk_methods:
            self.jamming_attacker = JammingAttacker(atker=atker, atker_ids=self.atker_ids, **kwargs)
        if 'replay' in atk_methods:
            self.replay_attacker = ReplayAttacker(atker=atker, atker_ids=self.atker_ids, **kwargs)
        if 'spoofing' in atk_methods:
            perceptual_attacker_prompt_template = kwargs["perceptual_attacker_prompt_template"]
            self.perceptual_attacker = PerceptualAttacker(
                atker=atker,
                atker_ids=self.atker_ids,
                prompt_template=perceptual_attacker_prompt_template,
                **kwargs,
            )
            self.action_attacker = ActionAttacker(atker=atker, atker_ids=self.atker_ids, **kwargs)
        if 'sybil' in atk_methods:
            self.sybil_attacker = SybilAttacker(atker=atker, atker_ids=self.atker_ids, **kwargs)
        
        

    def _initialize_defenders(self, defender, defense_methods, **kwargs):
        if 'firewall' in defense_methods:
            self.firewall_defender = FirewallDefender(defender=defender, **kwargs)
        else:
            self.firewall_defender = None
        if 'lpc' in defense_methods:
            self.lpc_defender = LPConsistencyDefender(defender=defender, **kwargs)
        else:
            self.lpc_defender = None
        if 'msc' in defense_methods:
            self.msc_defender = MSConsensusDefender(defender=defender, **kwargs)
        else:
            self.msc_defender = None
            

    def simulate_attack(self, message, self_message, ego_idx):
        """Run all attacker stages only when the ego is the self vehicle."""
        if ego_idx != self.self_id:
            return message
        ######## jamming ########
        if getattr(self, "jamming_attacker", None) is not None:
            message = self.jamming_attacker.attack(message, self_message, ego_idx)
        ######## replay ########
        if getattr(self, "replay_attacker", None) is not None:
            message = self.replay_attacker.attack(message, self_message, ego_idx)  
        ######## spoofing (perceptual + action) ########
        if getattr(self, "perceptual_attacker", None) is not None:
            message = self.perceptual_attacker.attack(message, self_message, ego_idx)
        if getattr(self, "action_attacker", None) is not None:
            message = self.action_attacker.attack(message, self_message, ego_idx)
        ######## sybil ########
        if getattr(self, "sybil_attacker", None) is not None:
            message = self.sybil_attacker.attack(message, self_message, ego_idx)
        return message

    def simulate_defense(self, message, ego_idx, **kwargs):
        """
        Run defense in sync or async (blocked) mode.
        Returns: (message, malicious_ids: set[int])

        Populates self.last_defense_timing with timings only for initialized defenders.
        """
        # If not our ego, skip defense but keep a consistent timing shape
        if ego_idx != self.self_id:
            self.last_defense_timing = {
                "total_s": 0.0,
                "firewall_s": 0.0,
                "lpc_s": 0.0,
                "msc_s": 0.0,
            }
            return message, set()

        # Async path delegated to the async helper (but still blocked)
        if self.defender_async_mode:
            msg, malicious_ids = run_coro_blocking(self._simulate_defense_async(message, ego_idx, **kwargs))
            msg = [m for m in msg if m["idx"] not in malicious_ids]
            return msg, malicious_ids

        # Sync path
        active = []  # list of (name, instance, call_fn)
        if getattr(self, "firewall_defender", None) is not None:
            active.append(("firewall", self.firewall_defender, "defend"))
        if getattr(self, "lpc_defender", None) is not None:
            active.append(("lpc", self.lpc_defender, "defend"))
        if getattr(self, "msc_defender", None) is not None:
            active.append(("msc", self.msc_defender, "defend"))

        if not active:
            # No defenders configured; return as-is
            self.last_defense_timing = {
                "total_s": 0.0,
                "firewall_s": 0.0,
                "lpc_s": 0.0,
                "msc_s": 0.0,
            }
            return message, set()

        t_total0 = time.perf_counter()
        per_t = {"firewall_s": 0.0, "lpc_s": 0.0, "msc_s": 0.0}

        results = []  # each item: (name, result)
        for name, inst, fn in active:
            t0 = time.perf_counter()
            res = getattr(inst, fn)(deepcopy(message), ego_idx, **kwargs)
            elapsed = time.perf_counter() - t0
            per_t[f"{name}_s"] = elapsed
            results.append((name, res))

        malicious_ids = set()
        if self.trust_score_system:
            # Each res is a dict[int->score]; average over available defenders only
            dicts = [res for _, res in results]
            all_ids = set()
            for d in dicts:
                all_ids.update(d.keys())
            denom = max(len(dicts), 1)
            avg_scores = {i: sum(d.get(i, 1.0) for d in dicts) / denom for i in all_ids}
            malicious_ids = {i for i, s in avg_scores.items() if s >= self.trust_score_threshold}
        else:
            for _, res in results:
                malicious_ids |= set(res)

        total = time.perf_counter() - t_total0
        self.pred_malicious_ids.append(list(malicious_ids))

        self.last_defense_timing = {
            "total_s": total,
            **per_t,
        }

        print(f"Defense completed in {total:.3f}s:")
        if per_t["firewall_s"]:
            print(f"  Firewall: {per_t['firewall_s']:.3f}s")
        if per_t["lpc_s"]:
            print(f"  LPC: {per_t['lpc_s']:.3f}s")
        if per_t["msc_s"]:
            print(f"  MSC: {per_t['msc_s']:.3f}s")

        # Detailed sub-timings only for initialized defenders
        self.last_defense_timing_detail = {
            "firewall": getattr(self.firewall_defender, "get_last_timing", lambda: None)() if self.firewall_defender else None,
            "lpc": getattr(self.lpc_defender, "get_last_timing", lambda: None)() if self.lpc_defender else None,
            "msc": getattr(self.msc_defender, "get_last_timing", lambda: None)() if self.msc_defender else None,
            "summary": self.last_defense_timing,
        }

        message = [msg for msg in message if msg["idx"] not in malicious_ids]
        return message, malicious_ids


    async def _simulate_defense_async(self, message, ego_idx, **kwargs):
        """Async defense: run only initialized defenders concurrently and merge results."""
        if ego_idx != self.self_id:
            self.last_defense_timing = {
                "total_s": 0.0, "firewall_s": 0.0, "lpc_s": 0.0, "msc_s": 0.0
            }
            return message, set()

        # Build active async tasks
        active = []
        if getattr(self, "firewall_defender", None) is not None:
            active.append(("firewall", self.firewall_defender))
        if getattr(self, "lpc_defender", None) is not None:
            active.append(("lpc", self.lpc_defender))
        if getattr(self, "msc_defender", None) is not None:
            active.append(("msc", self.msc_defender))

        if not active:
            self.last_defense_timing = {
                "total_s": 0.0, "firewall_s": 0.0, "lpc_s": 0.0, "msc_s": 0.0
            }
            return message, set()

        async def _timed(name, coro):
            t0 = time.perf_counter()
            try:
                res = await coro
                return name, res, (time.perf_counter() - t0), None
            except Exception as e:
                return name, None, (time.perf_counter() - t0), e

        t_total0 = time.perf_counter()

        tasks = [
            _timed(name, defender.defend_async(deepcopy(message), ego_idx, **kwargs))
            for name, defender in active
        ]
        timed_results = await asyncio.gather(*tasks, return_exceptions=False)

        malicious_ids = set()
        timing_map = {"firewall_s": 0.0, "lpc_s": 0.0, "msc_s": 0.0}

        if self.trust_score_system:
            dicts = []
            for name, res, elapsed, err in timed_results:
                timing_map[f"{name}_s"] = elapsed
                if err is not None:
                    print(f"Warning: {name} defense task failed: {err}")
                    continue
                dicts.append(res)
            all_ids = set()
            for d in dicts:
                all_ids.update(d.keys())
            denom = max(len(dicts), 1)
            avg_scores = {i: sum(d.get(i, 1.0) for d in dicts) / denom for i in all_ids}
            malicious_ids = {i for i, s in avg_scores.items() if s >= self.trust_score_threshold}
        else:
            for name, res, elapsed, err in timed_results:
                timing_map[f"{name}_s"] = elapsed
                if err is not None:
                    print(f"Warning: {name} defense task failed: {err}")
                    continue
                malicious_ids |= set(res)

        total = time.perf_counter() - t_total0
        self.pred_malicious_ids.append(list(malicious_ids))
        self.last_defense_timing = {"total_s": total, **timing_map}

        print(f"Defense completed in {total:.3f}s:")
        if timing_map["firewall_s"]:
            print(f"  Firewall: {timing_map['firewall_s']:.3f}s")
        if timing_map["lpc_s"]:
            print(f"  LPC: {timing_map['lpc_s']:.3f}s")
        if timing_map["msc_s"]:
            print(f"  MSC: {timing_map['msc_s']:.3f}s")

        self.last_defense_timing_detail = {
            "firewall": getattr(self.firewall_defender, "get_last_timing", lambda: None)() if getattr(self, "firewall_defender", None) else None,
            "lpc": getattr(self.lpc_defender, "get_last_timing", lambda: None)() if getattr(self, "lpc_defender", None) else None,
            "msc": getattr(self.msc_defender, "get_last_timing", lambda: None)() if getattr(self, "msc_defender", None) else None,
            "summary": self.last_defense_timing,
        }
        return message, malicious_ids


    def clean_up(self):
        """Reset recorded predictions."""
        self.pred_malicious_ids = []


    def evaluate(self, gamma=0.95, lam=1.0, eps=1e-9):
        """
        Compute evaluation metrics over the stored predictions.
        See doc/eval_metric.md for details.
        """
        
        atk_idx = self.atker_ids
        pred = self.pred_malicious_ids
        N = self.ego_num
        if self.with_sybil:
            sybil_num = self.sybil_num * len(self.atker_ids)
            atk_idx = atk_idx + [N + i for i in range(sybil_num)]
            N = N + sybil_num
        import pdb; pdb.set_trace()
        A = set(atk_idx)
        k = len(A)
        T = len(pred)

        F1s, Jaccs, ws = [], [], []
        hat = {i: [0] * T for i in range(N)}

        for t, P in enumerate(pred, start=1):
            P = set(P)
            TP = len(P & A)
            FP = len(P - A)
            FN = len(A - P)
            prec = TP / (TP + FP + eps)
            rec = TP / (TP + FN + eps)
            f1 = 2 * prec * rec / (prec + rec + eps)
            jacc = TP / (len(P | A) + eps)
            F1s.append(f1)
            Jaccs.append(jacc)
            ws.append(gamma ** (t - 1))
            for i in P:
                hat[i][t - 1] = 1

        F1_mean = sum(F1s) / max(T, 1)
        J_mean = sum(Jaccs) / max(T, 1)
        WF1 = sum(w * f for w, f in zip(ws, F1s)) / (sum(ws) + eps)
        WJacc = sum(w * j for w, j in zip(ws, Jaccs)) / (sum(ws) + eps)

        # LADS
        tau = {}
        for i in A:
            try:
                t_first = hat[i].index(1) + 1
                tau[i] = t_first
            except ValueError:
                tau[i] = math.inf

        c = [(1 - (t - 1) / max(T, 1)) if t != math.inf else 0.0 for t in tau.values()]
        normals = [j for j in range(N) if j not in A]
        b = [sum(hat[j]) / max(T, 1) for j in normals] if normals else [0.0]
        LADS = (sum(c) / max(k, 1)) - lam * (sum(b) / max(len(normals), 1))

        # stability
        flips = 0
        for i in range(N):
            seq = hat[i]
            flips += sum(int(seq[t] != seq[t - 1]) for t in range(1, T))
        FlipRate = flips / (N * max(T - 1, 1))

        # median tau among detected attackers
        finite_taus = sorted(t for t in tau.values() if t != math.inf)
        if finite_taus:
            median_tau = finite_taus[len(finite_taus) // 2]
        else:
            median_tau = float("inf")

        return {
            "F1_mean": F1_mean, "Jacc_mean": J_mean,
            "WF1": WF1, "WJacc": WJacc,
            "LADS": LADS, "FlipRate": FlipRate,
            "tau_stats": {
                "median": median_tau,
                "miss_rate": sum(1 for t in tau.values() if t == math.inf) / max(k, 1),
            },
        }
        
        
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


