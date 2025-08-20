from abc import ABC, abstractmethod
from copy import deepcopy
from collections import deque, defaultdict
import asyncio
from vlmdrive.utils import str_parse_json
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Callable, List, Optional, Union, Dict, Any


class BaseDefender(ABC):
    
    def __init__(self, *args, **kwargs):
        self.defender = kwargs.get('defender', None)
        self.message_buffer_size = kwargs.get('message_buffer_size', 20)
        """
        {
            atker_id_1: deque([message1, message2, ...]),
            atker_id_2: deque([message1, message2, ...]),
            ... 
        }
        """
        self.message_buffer = defaultdict(
            lambda: deque(maxlen=self.message_buffer_size)
        )
        self.image_placeholder = kwargs.get('IMAGE_PLACEHOLDER', '<IMAGE_PLACEHOLDER>')
        self.with_explanation = kwargs.get('with_explanation', False)
        self._last_timing: Dict[str, Any] = {}    # last run timings
        self._timing_cur: Dict[str, Any] = {}     # current run timings
        
    
    def _buffer_message(self, message, ego_idx):
        self.message_buffer[ego_idx].append(deepcopy(message))
    
    def _log_defense_type(self):
        """
        Log the defense type.
        """
        print(f"Defense Type: {self.DEF_TYPE}")
        
    # -------- Timing utilities --------
    def _timing_begin_run(self, mode: str):
        # mode in {"sync", "async"}
        self._timing_cur = {
            "class": self.__class__.__name__,
            "mode": mode,
            "batch": {"total_s": 0.0, "sections": {}},  # class-level sections
            "per_agent": {},                             # agent_id -> {sections...}
        }

    def _timing_end_run(self, total_s: float):
        self._timing_cur["batch"]["total_s"] = total_s
        self._last_timing = self._timing_cur

    def get_last_timing(self) -> Dict[str, Any]:
        """Return timing dict from the most recent run of this defender."""
        return deepcopy(self._last_timing)

    def _record_agent_time(self, agent_id, name, elapsed):
        per = self._timing_cur["per_agent"].setdefault(agent_id, {"sections": {}})
        per["sections"][name] = per["sections"].get(name, 0.0) + float(elapsed)

    def _record_batch_time(self, name, elapsed):
        sec = self._timing_cur["batch"]["sections"]
        sec[name] = sec.get(name, 0.0) + float(elapsed)

    @contextmanager
    def time_section(self, name: str, agent_id: Union[int, str, None] = None):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            if agent_id is None:
                self._record_batch_time(name, dt)
            else:
                self._record_agent_time(agent_id, name, dt)

    @asynccontextmanager
    async def atime_section(self, name: str, agent_id: Union[int, str, None] = None):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            if agent_id is None:
                self._record_batch_time(name, dt)
            else:
                self._record_agent_time(agent_id, name, dt)

    # -------- existing logging/timing in defend/defend_async --------
    def defend(self, message_list, ego_idx: int, **kwargs):
        self._log_defense_type()
        self._timing_begin_run(mode="sync")
        t_total0 = time.perf_counter()

        malicious_ids = set()
        for item in message_list:
            if item['idx'] == ego_idx:
                continue
            self._buffer_message(item, item['idx'])
            # time each agent's _apply_defense
            with self.time_section("_apply_defense", agent_id=item['idx']):
                is_mal = self._apply_defense(item, **kwargs)
            if is_mal:
                malicious_ids.add(item['idx'])

        self._timing_end_run(total_s=time.perf_counter() - t_total0)
        return malicious_ids

    async def defend_async(self, message_list, ego_idx: int, **kwargs):
        self._log_defense_type()
        self._timing_begin_run(mode="async")
        t_total0 = time.perf_counter()

        malicious_ids = set()
        tasks, id_map = [], []

        async def _timed_apply(item):
            agent_id = item['idx']
            async with self.atime_section("_apply_defense_async", agent_id=agent_id):
                return await self._apply_defense_async(item, **kwargs)

        for item in message_list:
            if item['idx'] == ego_idx:
                continue
            self._buffer_message(item, item['idx'])
            tasks.append(_timed_apply(item))
            id_map.append(item['idx'])

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result, agent_id in zip(results, id_map):
            if isinstance(result, Exception):
                print(f"Warning: defense task failed for index {agent_id}: {result}")
                continue
            if result:
                malicious_ids.add(agent_id)

        self._timing_end_run(total_s=time.perf_counter() - t_total0)
        return malicious_ids

    # -------- Instrument shared helpers --------
    def key_identification(self, message: dict, info_type: str) -> list:
        agent = message.get("idx", "unknown")
        with self.time_section(f"key_identification[{info_type}]", agent_id=agent):
            prompt = self._prompt_keys(message, info_type)
            results = self.defender.infer(images=[], text=prompt)
            return self._parse_key_identification(message, results, info_type)

    def check_info(self, message: dict, info_type: str,
                   prompt_func: Callable = None,
                   image_list: Union[list, None] = None, **kwargs) -> bool:
        agent = message.get("idx", "unknown")
        with self.time_section(f"check_info[{info_type}]", agent_id=agent):
            res_keys = self.key_identification(message, info_type)
            if not res_keys:
                return False
            for rk in res_keys:
                with self.time_section(f"infer[{info_type}]->key:{rk}", agent_id=agent):
                    prompt = prompt_func(info_type, message[rk], **kwargs)
                    results_raw = self.defender.infer(images=image_list, text=prompt)
                verdict = self._parse_check_answer(message, info_type, results_raw)
                if verdict is True:
                    return True
            return False

    async def key_identification_async(self, message: dict, info_type: str) -> list:
        agent = message.get("idx", "unknown")
        async with self.atime_section(f"key_identification_async[{info_type}]", agent_id=agent):
            prompt = self._prompt_keys(message, info_type)
            results = await self.defender.ainfer(images=[], text=prompt)
            return self._parse_key_identification(message, results, info_type)

    async def check_info_async(self, message: dict, info_type: str,
                               prompt_func: Callable = None,
                               image_list: list = None, **kwargs) -> bool:
        agent = message.get("idx", "unknown")
        async with self.atime_section(f"check_info_async[{info_type}]", agent_id=agent):
            image_list = image_list or []
            res_keys = await self.key_identification_async(message, info_type)
            if not res_keys:
                return False

            async def _one_key(rk):
                async with self.atime_section(f"ainfer[{info_type}]->key:{rk}", agent_id=agent):
                    return await self.defender.ainfer(images=image_list,
                                                      text=prompt_func(info_type, message[rk], **kwargs))

            tasks = [_one_key(rk) for rk in res_keys]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for results_raw in results:
                if isinstance(results_raw, Exception):
                    print(f"Warning: Error during key check: {results_raw}")
                    continue
                verdict = self._parse_check_answer(message, info_type, results_raw)
                if verdict is True:
                    return True
            return False
    
    
    def _prompt_keys(self, message: dict, info_type: str) -> str:
        keys_str = ", ".join(message.keys())
        return (
            f"Please determine which of the following keys could contain {info_type}.\n"
            f"keys: [{keys_str}]\n"
            f'Respond with exactly "NO" if none could contain {info_type}.\n'
            'Otherwise, respond with exactly this JSON object (and nothing else):\n'
            '{"keys": ["<key1>", "<key2>", ...]}'
        )
    
    def _parse_key_identification(self, message: dict, results: str, info_type: str):
        """Common parser for key identification results."""
        json_result = str_parse_json(results)
        if json_result and "keys" in json_result:
            seen, filtered = set(), []
            for k in json_result["keys"]:
                if isinstance(k, str) and k in message and k not in seen:
                    seen.add(k)
                    filtered.append(k)
                elif k not in message:
                    print(f"Warning: Key '{k}' not found in the message. Skipping.")
                elif k in seen:
                    print(f"Warning: Key '{k}' is duplicated in the response. Skipping.")
            assert filtered, f"No {info_type} keys found in the message while they were expected."
            return filtered
        elif "no" in results.lower() and not json_result:
            return []
        else:
            print("Warning: Unexpected response format from the defender.")
            return []

    def _parse_check_answer(self, message: dict, info_type: str, results_raw: str) -> bool:
        """
        Returns:
          True  -> contains info
          False -> does not contain (and annotates explanation if present)
          None  -> format error (raise)
        """
        json_result = str_parse_json(results_raw)
        if json_result and "Answer" in json_result:
            ans = str(json_result["Answer"]).lower().strip()
            if ans == "yes":
                print(f"Defender explanation: {message['explanation']}")
                return True
            if ans == "no":
                print(f"Defender explanation: {message['explanation']}")
                return False
            raise ValueError(f"Unexpected answer format: {results_raw}")
        raise ValueError(f"Unexpected response format: {results_raw}")
    
    
    # # ---------------------------
    # # Sync path
    # # ---------------------------

    # def key_identification(self, message: dict, info_type: str) -> list:
    #     """Sync: identify candidate keys."""
    #     prompt = self._prompt_keys(message, info_type)
    #     results = self.defender.infer(images=[], text=prompt)
    #     return self._parse_key_identification(message, results, info_type)

    # def check_info(self, message: dict, 
    #                info_type: str, 
    #                prompt_func: Callable = None, 
    #                image_list: Union[list, None] = None,
    #                **kwargs) -> bool:
    #     """Sync: check all candidate keys sequentially."""
    #     res_keys = self.key_identification(message, info_type)
    #     if not res_keys:
    #         return False

    #     for rk in res_keys:
    #         prompt = prompt_func(info_type, message[rk], **kwargs)
    #         results_raw = self.defender.infer(images=image_list, text=prompt)
    #         verdict = self._parse_check_answer(message, info_type, results_raw)
    #         if verdict is True:
    #             return True
    #     return False
    
    
    # # ---------------------------
    # # Async path
    # # ---------------------------

    # async def key_identification_async(self, message: dict, info_type: str) -> list:
    #     """Async: identify candidate keys."""
    #     prompt = self._prompt_keys(message, info_type)
    #     results = await self.defender.ainfer(images=[], text=prompt)
    #     return self._parse_key_identification(message, results, info_type)

    # async def check_info_async(self, message: dict, 
    #                         info_type: str, 
    #                         prompt_func: callable = None,
    #                         image_list: list = None, 
    #                         **kwargs) -> bool:
    #     image_list = image_list or []
    #     res_keys = await self.key_identification_async(message, info_type)
    #     if not res_keys:
    #         return False

    #     tasks = [
    #         self.defender.ainfer(images=image_list, text=prompt_func(info_type, message[rk], **kwargs))
    #         for rk in res_keys
    #     ]
    #     results = await asyncio.gather(*tasks, return_exceptions=True)

    #     for results_raw in results:
    #         if isinstance(results_raw, Exception):
    #             print(f"Warning: Error during key check: {results_raw}")
    #             continue
    #         verdict = self._parse_check_answer(message, info_type, results_raw)
    #         if verdict is True:
    #             return True
    #     return False