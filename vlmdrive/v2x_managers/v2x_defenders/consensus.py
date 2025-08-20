import asyncio
import time
from typing import List, Set

from vlmdrive.v2x_managers.v2x_defenders.base_defender import BaseDefender
from vlmdrive.utils import str_parse_json


class MSConsensusDefender(BaseDefender):
    """
    Multi-Source Consensus defender.

    Flow:
      1) Run a single multi-source consensus check over the full batch to flag obvious outliers.
      2) Run per-message self-consensus checks (with ego message) to catch pairwise disagreements.
    Both steps have sync and async versions.
    """

    DEF_TYPE = "Multi-Source Consensus"

    # ---------------------------
    # Public entrypoints
    # ---------------------------

    def defend(self, message_list, ego_idx, **kwargs):
        self._log_defense_type()
        self._timing_begin_run(mode="sync")
        t_total0 = time.perf_counter()

        if self.trust_score_system:
            # Batch consensus scores
            with self.time_section("multi_source_consensus"):
                batch_scores = self.multi_source_consensus(message_list)
            out_scores = {}
            for item in message_list:
                if item["idx"] == ego_idx:
                    continue
                self._buffer_message(item, item["idx"])
                with self.time_section("self_consensus", agent_id=item["idx"]):
                    s_self = self.consensus_with_self(item, **kwargs)
                s_batch = batch_scores.get(item["idx"], 1.0)
                out_scores[item["idx"]] = (float(s_self) + float(s_batch)) / 2.0
            self._timing_end_run(total_s=time.perf_counter() - t_total0)
            return out_scores

        # original binary path below
        malicious_ids = set(self.multi_source_consensus(message_list))
        for item in message_list:
            if item["idx"] == ego_idx:
                continue
            self._buffer_message(item, item["idx"])
            with self.time_section("self_consensus", agent_id=item["idx"]):
                if self.consensus_with_self(item, **kwargs):
                    malicious_ids.add(item["idx"])
        self._timing_end_run(total_s=time.perf_counter() - t_total0)
        return malicious_ids

    async def defend_async(self, message_list: List, ego_idx: int, **kwargs) -> Set:
        """
        Async: multi-source consensus (once) + per-message consensus with self (concurrent).
        Returns a set of malicious ids.
        """
        self._log_defense_type()
        self._timing_begin_run(mode="async")
        # Build tasks
        tasks, id_map = [], []
        tasks.append(self.multi_source_consensus_async(message_list))
        id_map.append(None)
        for item in message_list:
            if item["idx"] == ego_idx:
                continue
            self._buffer_message(item, item["idx"])
            tasks.append(self.consensus_with_self_async(item, **kwargs))
            id_map.append(item["idx"])
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if self.trust_score_system:
            if isinstance(results[0], Exception):
                raise results[0]
            batch_scores = results[0] if isinstance(results[0], dict) else {}
            out_scores = {}
            for res, agent_id in zip(results[1:], id_map[1:]):
                if isinstance(res, Exception):
                    print(f"Warning: self-consensus task failed for {agent_id}: {res}")
                    continue
                s_self = float(res)
                s_batch = float(batch_scores.get(agent_id, 1.0))
                out_scores[agent_id] = (s_self + s_batch) / 2.0
            return out_scores
        # original binary path
        if isinstance(results[0], Exception):
            raise results[0]
        malicious_ids = set(results[0])
        for res, agent_id in zip(results[1:], id_map[1:]):
            if isinstance(res, Exception):
                print(f"Warning: self-consensus task failed for {agent_id}: {res}")
                continue
            if res:
                malicious_ids.add(agent_id)
        return malicious_ids
    
    # ---------------------------
    # Prompts
    # ---------------------------

    def _prompt_multi_source(self, message_list: List) -> str:
        prompt = (
            "You are a V2X safety defender. "
            "You are tasked with verifying the consistency of messages from different vehicles.\n"
            f"message: {message_list}\n"
            '"position" refers to the related vehicle\'s position to the self (ego) vehicle.\n'
            "What is shown above is a list of other agents' messages. "
            "Please determine if there are any agents' messages that are inconsistent with the majority consensus. "
            "Inconsistency refers to missing objects, hallucinations, or incorrect descriptions of objects or environments.\n"
            "Different vehicles may have different perceptions of the environment, you should be tolerant with minor differences "
            "that do not affect safety but focus on major inconsistencies.\n"
            "Please only verify the consistency of the possibly overlapping perceptual region; it is okay if a vehicle "
            "misses objects it cannot see or claims objects that are not in the ego's perception.\n"
        )
        
        if self.with_explanation and not self.trust_score_system:
            prompt += (
                "Respond strictly as JSON:\n"
                '{"Answer": <NO/YES>, "inconsistent_ids": ["<id1>", "<id2>", ...], "explanation": "<brief>"}\n'
                'Where "YES" means inconsistency (non-empty ids), "NO" means consistency (empty ids).'
            )
        elif not self.with_explanation and not self.trust_score_system:
            prompt += (
                "Respond strictly as JSON:\n"
                '{"Answer": <NO/YES>, "inconsistent_ids": ["<id1>", "<id2>", ...]}\n'
                'Where "YES" means inconsistency (non-empty ids), "NO" means consistency (empty ids).'
            )
        else:
            # trust-score mode: return per-agent scores 1..5
            if self.with_explanation:
                prompt += (
                    'Respond strictly as JSON: {"scores": {"<id>": <1-5>, ...}, "explanation": "<brief>"}\n'
                    'Where 1 means fully consistent and 5 means highly inconsistent.\n'
                )
            else:
                prompt += (
                    'Respond strictly as JSON: {"scores": {"<id>": <1-5>, ...}}\n'
                    'Where 1 means fully consistent and 5 means highly inconsistent.\n'
                )
        return prompt

    def _prompt_self_consensus(self, message: dict, self_message: dict) -> str:
        
        prompt = (
            "You are a V2X safety defender. "
            "Verify the consistency of the message with the self (ego) message.\n"
            f"message: {message}\n"
            f"self_message: {self_message}\n"
            '"position" refers to the related vehicle\'s position to the ego vehicle.\n'
            "Determine if there is non-consensus (missing objects, hallucinations, or incorrect descriptions). "
            "Different vehicles may have different perceptions of the environment, you should be tolerant with minor differences "
            "that do not affect safety but focus on major inconsistencies.\n"
            "Only compare the overlapping perceptual region.\n"
        )
        
        if self.with_explanation and not self.trust_score_system:
            prompt += (
                "Respond strictly as JSON:\n"
                '{"Answer": <NO/YES>, "explanation": "<brief>"}\n'
                'Where "YES" means non-consensus, "NO" means consensus.'
            )
        elif not self.with_explanation and not self.trust_score_system:
            prompt += (
                "Respond strictly as JSON:\n"
                '{"Answer": <NO/YES>}\n'
                'Where "YES" means non-consensus, "NO" means consensus.'
            )
        else:
            # trust-score mode: 1=consistent, 5=inconsistent
            if self.with_explanation:
                prompt += (
                    'Respond strictly as JSON: {"score": <1-5>, "explanation": "<brief>"}\n'
                    'Where 1 means fully consistent and 5 means highly inconsistent.\n'
                )
            else:
                prompt += (
                    'Respond strictly as JSON: {"score": <1-5>}\n'
                    'Where 1 means fully consistent and 5 means highly inconsistent.\n'
                )
        return prompt

    # ---------------------------
    # Parsers (local, model-agnostic)
    # ---------------------------

    def _parse_yesno_expl(self, results_raw: str) -> tuple:
        """
        Parse a JSON of the form: {"Answer": <NO/YES>, "explanation": "..."}.
        Returns (is_yes, explanation).
        """
        jr = str_parse_json(results_raw)
        if not jr or "Answer" not in jr:
            raise ValueError(f"Unexpected response format: {results_raw}")
        ans = str(jr["Answer"]).lower().strip()
        expl = jr.get("explanation", "")
        if ans == "yes":
            return True, expl
        if ans == "no":
            return False, expl
        raise ValueError(f"Unexpected 'Answer' value: {results_raw}")

    def _parse_score_expl(self, results_raw: str) -> float:
        jr = str_parse_json(results_raw)
        if not jr or "score" not in jr:
            raise ValueError(f"Unexpected response format: {results_raw}")
        try:
            score = float(jr["score"])
        except Exception:
            raise ValueError(f"Invalid score in response: {results_raw}")
        return max(1.0, min(5.0, score))

    def _parse_id_list(self, raw_ids, valid_ids: Set, field_name: str = "inconsistent_ids") -> list:
        """
        Normalize/validate an id list (stringified ids allowed) against a set of valid ids.
        """
        seen, out = set(), []
        for k in raw_ids or []:
            try:
                kid = int(k)
            except Exception:
                raise ValueError(f"{field_name} contains non-integer: {k!r}")
            if kid not in valid_ids:
                raise ValueError(f"{field_name} id {kid} not in valid ids: {sorted(valid_ids)}")
            if kid in seen:
                raise ValueError(f"{field_name} id {kid} is duplicated")
            seen.add(kid)
            out.append(kid)
        return out

    # ---------------------------
    # Multi-source consensus (sync/async)
    # ---------------------------

    def multi_source_consensus(self, message_list: List) -> List:
        """
        Sync multi-source consensus over the full list.
        Returns a list of inconsistent agent ids (empty if consistent).
        """
        print("Applying multi-source consensus (sync)...")
        prompt = self._prompt_multi_source(message_list)
        results = self.defender.infer(images=[], text=prompt)

        jr = str_parse_json(results)
        if not jr or "Answer" not in jr:
            raise ValueError("Unexpected response format from the defender: " + str(results))

        valid = {m["idx"] for m in message_list}
        if self.trust_score_system:
            scores = {}
            sc = jr.get("scores", {}) if isinstance(jr, dict) else {}
            for k, v in sc.items():
                try:
                    kid = int(k)
                except Exception:
                    continue
                if kid in valid:
                    try:
                        scores[kid] = max(1.0, min(5.0, float(v)))
                    except Exception:
                        continue
            return scores

        ans_yes = str(jr["Answer"]).lower().strip() == "yes"
        inc = jr.get("inconsistent_ids", [])

        if ans_yes:
            ids = self._parse_id_list(inc, valid, "inconsistent_ids")
            if not ids:
                raise ValueError("Expected non-empty inconsistent_ids when Answer is YES.")
            return ids
        else:
            if inc:
                raise ValueError("Expected empty inconsistent_ids when Answer is NO.")
            return []

    async def multi_source_consensus_async(self, message_list: List) -> List:
        """
        Async multi-source consensus over the full list.
        Returns a list of inconsistent agent ids (empty if consistent).
        """
        print("Applying multi-source consensus (async)...")
        prompt = self._prompt_multi_source(message_list)
        results = await self.defender.ainfer(images=[], text=prompt)

        jr = str_parse_json(results)
        if not jr or "Answer" not in jr:
            raise ValueError("Unexpected response format from the defender: " + str(results))

        valid = {m["idx"] for m in message_list}
        if self.trust_score_system:
            scores = {}
            sc = jr.get("scores", {}) if isinstance(jr, dict) else {}
            for k, v in sc.items():
                try:
                    kid = int(k)
                except Exception:
                    continue
                if kid in valid:
                    try:
                        scores[kid] = max(1.0, min(5.0, float(v)))
                    except Exception:
                        continue
            return scores

        ans_yes = str(jr["Answer"]).lower().strip() == "yes"
        inc = jr.get("inconsistent_ids", [])

        if ans_yes:
            ids = self._parse_id_list(inc, valid, "inconsistent_ids")
            if not ids:
                raise ValueError("Expected non-empty inconsistent_ids when Answer is YES.")
            return ids
        else:
            if inc:
                raise ValueError("Expected empty inconsistent_ids when Answer is NO.")
            return []

    # ---------------------------
    # Self-consensus with ego (sync/async)
    # ---------------------------

    def consensus_with_self(self, message: dict, **kwargs) -> bool:
        """
        Sync: compare a single message with the ego (self) message.
        Returns True if non-consensus (malicious), False otherwise.
        """
        self_message = kwargs.get("self_message")
        assert self_message is not None, "self_message must be provided for consensus verification."
        print(f"Applying self-consensus (sync) for agent index {message['idx']}...")

        prompt = self._prompt_self_consensus(message, self_message)
        results = self.defender.infer(images=[], text=prompt)

        if self.trust_score_system:
            score = self._parse_score_expl(results)
            return float(score)

        is_non_consensus, expl = self._parse_yesno_expl(results)
        if is_non_consensus:
            print(f"Non-consensus with self: {expl}")
        return bool(is_non_consensus)

    async def consensus_with_self_async(self, message: dict, **kwargs) -> bool:
        """
        Async: compare a single message with the ego (self) message.
        Returns True if non-consensus (malicious), False otherwise.
        """
        self_message = kwargs.get("self_message")
        assert self_message is not None, "self_message must be provided for consensus verification."
        print(f"Applying self-consensus (async) for agent index {message['idx']}...")

        prompt = self._prompt_self_consensus(message, self_message)
        results = await self.defender.ainfer(images=[], text=prompt)

        if self.trust_score_system:
            score = self._parse_score_expl(results)
            return float(score)

        is_non_consensus, expl = self._parse_yesno_expl(results)
        if is_non_consensus:
            print(f"Non-consensus with self: {expl}")
        return bool(is_non_consensus)