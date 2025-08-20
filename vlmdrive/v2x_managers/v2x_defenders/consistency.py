import asyncio
from copy import deepcopy
from vlmdrive.v2x_managers.v2x_defenders.base_defender import BaseDefender
from vlmdrive.utils import str_parse_json


class LPConsistencyDefender(BaseDefender):
    """
    Language-Perception Consistency Verification with both sync and async modes.
    """

    DEF_TYPE = "Language-Perception Consistency Verification"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Provide a default placeholder in case upstream doesn't set it
        self.image_placeholder = getattr(self, "image_placeholder", "<IMAGE_PLACEHOLDER>")

    # ---------------------------------------------------------------------
    # Shared helpers (prompt builders + parsers)
    # ---------------------------------------------------------------------

    def _prompt_ego_lpc(self, info_text: str, lang_desc, **kwargs) -> str:
        # rel_pos can be numpy array or list; ensure printable
        
        if "rel_pos" not in kwargs:
            raise KeyError("Missing 'rel_pos' in kwargs. Provide the relative position.")
        rel_pos = kwargs["rel_pos"]
        rel_pos_str = rel_pos.tolist() if hasattr(rel_pos, "tolist") else rel_pos
        
        prompt = (
            "The following image is the front view of the ego vehicle's perception "
            f"{self.image_placeholder}\n"
            f"The following message is a language description of other vehicles with the relative "
            f"position {rel_pos_str}.\n"
            f"message: {lang_desc}\n"
            "Please determine if there is any inconsistency between the following language description "
            "from other vehicles with the ego vehicle's perception.\n"
            "Inconsistency refers to missing objects, hallucinations, or incorrect descriptions of "
            "objects or environments.\n"
            "Please only verify the consistency of the possibly overlapping perceptual region; it is "
            "acceptable if the sender misses objects outside its view or mentions objects not visible "
            "to the ego.\n"
        )
        
        if self.with_explanation:
            prompt += (
                "Your response should be a JSON object containing an answer and a brief explanation that "
                'strictly follows the format:\n'
                '{"Answer": <NO/YES>, "explanation": "<brief explanation>"}\n'
            )
        else:
            prompt += (
                "Your response should be a JSON object containing only the answer that strictly follows "
                'the format: {"Answer": <NO/YES>}\n'
            )
        return prompt


    # ---------------------------------------------------------------------
    # Actors LPC (sync + async) - placeholder for future perceptual checks
    # ---------------------------------------------------------------------

    def actors_lpc(self, message):
        """
        Sync placeholder for actor-level LPC.
        Returns (message, is_malicious).
        """
        # Not implemented (no perceptual data available in this version)
        return False

    async def actors_lpc_async(self, message):
        """
        Async placeholder for actor-level LPC.
        Returns (message, is_malicious).
        """
        return False
    

    # ---------------------------------------------------------------------
    # Apply defense (sync + async) – used by inherited defend/defend_async
    # ---------------------------------------------------------------------

    def _apply_defense(self, message, **kwargs):
        """
        Sync: apply language-perception consistency verification.
        Returns is_malicious.
        """
        print(f"Applying LPC verification (sync) for agent index {message['idx']}...")
        is_mal_actors = self.actors_lpc(message)
        front_img = kwargs.get("front_image_ego")
        assert front_img is not None, "front_image_ego is required for LPC."
        is_mal_ego = self.check_info(message,
                                     "perceptual information",
                                     prompt_func=self._prompt_ego_lpc,
                                     image_list=[front_img],
                                     rel_pos=message.get("position"))
        is_malicious = is_mal_actors or is_mal_ego
        return is_malicious

    async def _apply_defense_async(self, message, **kwargs):
        """
        Async: apply language-perception consistency verification concurrently.
        Returns is_malicious.
        """
        print(f"Applying LPC verification (async) for agent index {message['idx']}...")
        task_actors = self.actors_lpc_async(message)
        task_ego = self.check_info_async(
            message,
            "perceptual information",
            prompt_func=self._prompt_ego_lpc,
            image_list=[kwargs.get("front_image_ego")],
            rel_pos=message.get("position")
        )
        results = await asyncio.gather(task_actors, task_ego, return_exceptions=True)
        is_malicious = False
        for res in results:
            if isinstance(res, Exception):
                continue
            if res:
                is_malicious = True
                break
        return is_malicious