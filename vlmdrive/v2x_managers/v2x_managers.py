from vlmdrive.v2x_managers.v2x_attackers.perceptual_attacker import PerceptualAttacker
from vlmdrive.v2x_managers.v2x_attackers.action_attacker import ActionAttacker
from vlmdrive.v2x_managers.v2x_attackers.comm_attacker import CommAttacker

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
        self.trust_score_threshold = defender_config.get("trust_score_threshold", 4.0)


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
            api_model_name=atker_config["api_model_name"],
            api_base_url=atker_config["api_base_url"],
            api_key=atker_config["api_key"],
            image_placeholder=atker_config["IMAGE_PLACEHOLDER"],  # <== use attacker's placeholder
            async_mode=self.atker_aysnc_mode,
        )
        atker = atker_helpers["atker"]

        defender_helpers = configure_vlm_helpers(
            name=defender_config["name"],
            api_model_name=defender_config["api_model_name"],
            api_base_url=defender_config["api_base_url"],
            api_key=defender_config["api_key"],
            image_placeholder=defender_config["IMAGE_PLACEHOLDER"],
            async_mode=self.defender_async_mode,
        )
        defender = defender_helpers["defender"]

        self._initialize_attackers(
            atker,
            message_buffer_size=atker_config.get("message_buffer_size", 20),
            IMAGE_PLACEHOLDER=atker_config["IMAGE_PLACEHOLDER"],
        )
        self._initialize_defenders(
            defender,
            take_malicious=defender_config.get("take_malicious", False),
            message_buffer_size=defender_config.get("message_buffer_size", 20),
            IMAGE_PLACEHOLDER=defender_config["IMAGE_PLACEHOLDER"],
            with_explanation=defender_config.get("with_explanation", False),
            trust_score_system=defender_config.get("trust_score_system", True),
        )

    def _initialize_attackers(self, atker, **kwargs):
        self.perceptual_attacker = PerceptualAttacker(atker=atker, **kwargs)
        self.action_attacker = ActionAttacker(atker=atker, **kwargs)
        self.comm_attacker = CommAttacker(atker=atker, **kwargs)

    def _initialize_defenders(self, defender, **kwargs):
        self.firewall_defender = FirewallDefender(defender=defender, **kwargs)
        self.lpc_defender = LPConsistencyDefender(defender=defender, **kwargs)
        self.msc_defender = MSConsensusDefender(defender=defender, **kwargs)

    def simulate_attack(self, message, ego_idx):
        """Run all attacker stages only when the ego is the self vehicle."""
        if ego_idx != self.self_id:
            return message
        msg = self.perceptual_attacker.attack(message, ego_idx)
        msg = self.action_attacker.attack(msg, ego_idx)
        msg = self.comm_attacker.attack(msg, ego_idx)
        return msg

    def simulate_defense(self, message, ego_idx, **kwargs):
        """
        Run defense in sync or async (blocked) mode.
        Returns: (message, malicious_ids: set[int])

        Populates self.last_defense_timing with:
          {
            "total_s": <float>,
            "firewall_s": <float>,
            "lpc_s": <float>,
            "msc_s": <float>
          }
        """

        # ---- sync path ----
        if ego_idx != self.self_id:
            # no defense needed when ego isn't self; still record a trivial timing
            self.last_defense_timing = {
                "total_s": 0.0, "firewall_s": 0.0, "lpc_s": 0.0, "msc_s": 0.0
            }
            return message, set()
        
        if self.defender_async_mode:
            message, malicious_ids = run_coro_blocking(self._simulate_defense_async(message, ego_idx, **kwargs))
            message = [msg for msg in message if msg["idx"] not in malicious_ids]
            return message, malicious_ids

        malicious_ids = set()

        t_total0 = time.perf_counter()

        # Firewall
        t0 = time.perf_counter()
        mal_firewall = self.firewall_defender.defend(deepcopy(message), ego_idx, **kwargs)
        t_firewall = time.perf_counter() - t0

        # LPC
        t0 = time.perf_counter()
        mal_lpc = self.lpc_defender.defend(deepcopy(message), ego_idx, **kwargs)
        t_lpc = time.perf_counter() - t0

        # MSC
        t0 = time.perf_counter()
        mal_msc = self.msc_defender.defend(deepcopy(message), ego_idx, **kwargs)
        t_msc = time.perf_counter() - t0

        if self.trust_score_system:
            # each is a dict[int->score]
            all_ids = {i for d in (mal_firewall, mal_lpc, mal_msc) for i in d.keys()}
            avg_scores = {i: (
                (mal_firewall.get(i, 1.0) + mal_lpc.get(i, 1.0) + mal_msc.get(i, 1.0)) / 3.0
            ) for i in all_ids}
            # threshold to derive malicious_ids for legacy metrics
            malicious_ids = {i for i, s in avg_scores.items() if s >= self.trust_score_threshold}
        else:
            malicious_ids |= mal_firewall
            malicious_ids |= mal_lpc
            malicious_ids |= mal_msc

        total = time.perf_counter() - t_total0

        # record for evaluation
        self.pred_malicious_ids.append(list(malicious_ids))

        # store timing
        self.last_defense_timing = {
            "total_s": total,
            "firewall_s": t_firewall,
            "lpc_s": t_lpc,
            "msc_s": t_msc,
        }
        print(f"Defense completed in {total:.3f}s: ")
        print(f"  Firewall: {t_firewall:.3f}s")
        print(f"  LPC: {t_lpc:.3f}s")
        print(f"  MSC: {t_msc:.3f}s")
        
        self.last_defense_timing_detail = {
            "firewall": self.firewall_defender.get_last_timing(),
            "lpc": self.lpc_defender.get_last_timing(),
            "msc": self.msc_defender.get_last_timing(),
            "summary": self.last_defense_timing,  # your overall totals
        }
        pprint(self.last_defense_timing_detail)
        
        message = [msg for msg in message if msg["idx"] not in malicious_ids]

        return message, malicious_ids

    async def _simulate_defense_async(self, message, ego_idx, **kwargs):
        """
        Async defense: run all defenders concurrently and merge IDs.

        Populates self.last_defense_timing with:
          {
            "total_s": <float>,
            "firewall_s": <float>,
            "lpc_s": <float>,
            "msc_s": <float>
          }
        """
        if ego_idx != self.self_id:
            # no defense needed when ego isn't self; still record a trivial timing
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

        # launch all three defenders concurrently, each timed
        tasks = [
            _timed("firewall", self.firewall_defender.defend_async(deepcopy(message), ego_idx, **kwargs)),
            _timed("lpc",      self.lpc_defender.defend_async(deepcopy(message), ego_idx, **kwargs)),
            _timed("msc",      self.msc_defender.defend_async(deepcopy(message), ego_idx, **kwargs)),
        ]
        timed_results = await asyncio.gather(*tasks, return_exceptions=False)

        malicious_ids = set()
        timing_map = {"firewall_s": 0.0, "lpc_s": 0.0, "msc_s": 0.0}

        if self.trust_score_system:
            # results are dicts
            dicts = []
            for name, res, elapsed, err in timed_results:
                timing_map[f"{name}_s"] = elapsed
                if err is not None:
                    print(f"Warning: {name} defense task failed: {err}")
                    continue
                dicts.append(res)
            # merge and average across defenders
            all_ids = set()
            for d in dicts:
                all_ids.update(d.keys())
            avg_scores = {i: sum(d.get(i, 1.0) for d in dicts) / max(len(dicts), 1) for i in all_ids}
            malicious_ids = {i for i, s in avg_scores.items() if s >= self.trust_score_threshold}
        else:
            for name, res, elapsed, err in timed_results:
                timing_map[f"{name}_s"] = elapsed
                if err is not None:
                    print(f"Warning: {name} defense task failed: {err}")
                    continue
                malicious_ids |= set(res)

        total = time.perf_counter() - t_total0

        # record for evaluation
        self.pred_malicious_ids.append(list(malicious_ids))

        # store timing
        self.last_defense_timing = {
            "total_s": total,
            **timing_map,
        }
        
        print(f"Defense completed in {total:.3f}s: ")
        print(f"  Firewall: {timing_map['firewall_s']:.3f}s")
        print(f"  LPC: {timing_map['lpc_s']:.3f}s")
        print(f"  MSC: {timing_map['msc_s']:.3f}s")
        
        self.last_defense_timing_detail = {
            "firewall": self.firewall_defender.get_last_timing(),
            "lpc": self.lpc_defender.get_last_timing(),
            "msc": self.msc_defender.get_last_timing(),
            "summary": self.last_defense_timing,
        }
        
        pprint(self.last_defense_timing_detail)
        
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


