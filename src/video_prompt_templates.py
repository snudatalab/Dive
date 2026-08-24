from dataclasses import dataclass


@dataclass(frozen=True)
class VideoPromptParts:
    system_prefix: str
    user_prefix_before_video: str
    video_placeholder: str
    suffix_after_video_before_query: str
    postfix_after_query: str
    followup_user_prefix: str
    followup_postfix_after_query: str | None = None
    image_placeholder: str | None = None

    @property
    def prefix_before_ctx(self) -> str:
        return self.system_prefix + self.user_prefix_before_video

    def _get_visual_placeholder(
        self,
        num_images: int | None = None,
        visual_mode: str = "video",
    ) -> str:
        if visual_mode == "images":
            if self.image_placeholder is None:
                raise ValueError("image placeholder is not registered for this model")
            if num_images is None or num_images <= 0:
                raise ValueError("num_images must be > 0 when visual_mode='images'")
            return self.image_placeholder * num_images
        return self.video_placeholder

    def render_prefill_prompt(
        self,
        num_images: int | None = None,
        visual_mode: str = "video",
    ) -> str:
        visual_placeholder = self._get_visual_placeholder(
            num_images=num_images,
            visual_mode=visual_mode,
        )
        return (
            f"{self.prefix_before_ctx}"
            f"{visual_placeholder}"
            f"{self.suffix_after_video_before_query}"
        )

    def render_query_tail(
        self,
        query: str,
        num_images: int | None = None,
        visual_mode: str = "video",
    ) -> str:
        full_prompt = self.render_prompt(
            query,
            num_images=num_images,
            visual_mode=visual_mode,
        )
        prefill_prompt = self.render_prefill_prompt(
            num_images=num_images,
            visual_mode=visual_mode,
        )
        if not full_prompt.startswith(prefill_prompt):
            raise ValueError("full prompt does not start with the expected prefill prompt")
        return full_prompt[len(prefill_prompt):]

    def render_prompt(
        self,
        query: str,
        num_images: int | None = None,
        visual_mode: str = "video",
    ) -> str:
        query = query.strip()
        visual_placeholder = self._get_visual_placeholder(
            num_images=num_images,
            visual_mode=visual_mode,
        )
        return (
            f"{self.prefix_before_ctx}"
            f"{visual_placeholder}"
            f"{self.suffix_after_video_before_query}"
            f"{query}"
            f"{self.postfix_after_query}"
        )

    def render_followup_query(self, query: str) -> str:
        query = query.strip()
        followup_postfix = self.postfix_after_query if self.followup_postfix_after_query is None else self.followup_postfix_after_query
        return f"{self.followup_user_prefix}{query}{followup_postfix}"

    def render_answer_with_followups(
        self,
        answer: str,
        followup_pairs: list[tuple[str, str]],
    ) -> str:
        rendered = str(answer).strip()
        for query, followup_answer in followup_pairs:
            rendered += self.render_followup_query(query)
            rendered += str(followup_answer).strip()
        return rendered


_QWEN25_VL = VideoPromptParts(
    system_prefix="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    user_prefix_before_video="<|im_start|>user\n<|vision_start|>",
    video_placeholder="<|video_pad|>",
    suffix_after_video_before_query="<|vision_end|>",
    postfix_after_query="<|im_end|>\n<|im_start|>assistant\n",
    followup_user_prefix="<|im_end|>\n<|im_start|>user\n",
)

_QWEN3_VL = VideoPromptParts(
    system_prefix="",
    user_prefix_before_video="<|im_start|>user\n<|vision_start|>",
    video_placeholder="<|video_pad|>",
    suffix_after_video_before_query="<|vision_end|>",
    postfix_after_query="<|im_end|>\n<|im_start|>assistant\n",
    followup_user_prefix="<|im_end|>\n<|im_start|>user\n",
)

_INTERNVL3 = VideoPromptParts(
    system_prefix="",
    user_prefix_before_video="<|im_start|>user\n",
    video_placeholder="<video>\n",
    suffix_after_video_before_query="",
    postfix_after_query="<|im_end|>\n<|im_start|>assistant\n",
    followup_user_prefix="<|im_end|>\n<|im_start|>user\n",
    image_placeholder="<IMG_CONTEXT>\n",
)

_LLAVA_ONEVISION = VideoPromptParts(
    system_prefix="",
    user_prefix_before_video="<|im_start|>user ",
    video_placeholder="<video>",
    suffix_after_video_before_query="\n",
    postfix_after_query="<|im_end|><|im_start|>assistant\n",
    followup_user_prefix="<|im_end|><|im_start|>user \n",
    followup_postfix_after_query="<|im_end|><|im_start|>assistant \n",
    image_placeholder="<image>",
)

_LLAVA_ONEVISION_15 = VideoPromptParts(
    system_prefix="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    user_prefix_before_video="<|im_start|>user\n<|vision_start|>",
    video_placeholder="<|video_pad|>",
    suffix_after_video_before_query="<|vision_end|>",
    postfix_after_query="<|im_end|>\n<|im_start|>assistant\n",
    followup_user_prefix="<|im_end|>\n<|im_start|>user\n",
)


def get_video_prompt_parts(model_name: str) -> VideoPromptParts:
    model_name = model_name.lower()

    if "llava-onevision-1.5" in model_name:
        return _LLAVA_ONEVISION_15
    if "qwen2.5-vl" in model_name:
        return _QWEN25_VL
    if "qwen3-vl" in model_name:
        return _QWEN3_VL
    if "internvl3" in model_name:
        return _INTERNVL3
    if "llava-onevision" in model_name:
        return _LLAVA_ONEVISION

    raise ValueError(f"No manual video prompt template registered for model '{model_name}'.")
