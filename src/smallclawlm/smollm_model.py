"""SmolLMModel - Local SmolLM3-3B executor for the slow path.

Uses llama-cpp-python (GGUF) for zero-external-token inference.
SmolLM3 natively supports python_tools mode outputting <code>tool()</code>
blocks that smolagents CodeAgent expects.

Architecture:
  Fast path:  user input → router → Pipeline → direct API call → result
  Slow path:  user input → router → NLMAgent → SmolLMModel.generate()
              → <code>tool_call()</code> → CodeAgent executes → result

SmolLM3 modes (set via system prompt):
  - /no_think (default): faster, higher BFCL (92.3%), no reasoning traces
  - /think: reasoning traces before answers, lower BFCL (88.8%)

Workaround: SmolLM3's GGUF embeds a {% generation %} jinja2 tag that
llama-cpp-python can't parse. We catch the error and skip the broken
template registration, using raw completion API instead.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

logger = logging.getLogger(__name__)

# Default model path
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "smallclawlm" / "models"
DEFAULT_MODEL_FILE = "smollm3-3b-q4_k_m.gguf"

# /no_think mode system prompt with python_tools instructions
SYSTEM_PROMPT = """<|im_start|>system
/nosplit

You are a tool-calling assistant. You MUST output Python code inside <code> tags.

Rules:
1. When you need to call a tool, output EXACTLY: <code>tool_name(args)</code>
2. When done, output: <code>final_answer(your_answer)</code>
3. Do NOT output any other format — no markdown, no ```python```, no plain text.
4. Only call tools that are listed below.
5. One tool call per <code> block.

Available tools:"""


def _load_llm_bypassing_chat_template(model_path, n_ctx, n_gpu_layers, n_threads):
    """Load a GGUF model, bypassing SmolLM3's broken {% generation %} jinja2 template."""
    from llama_cpp import Llama
    import llama_cpp.llama_chat_format as lcf

    _orig_formatter_init = lcf.Jinja2ChatFormatter.__init__

    def _safe_formatter_init(self_fmt, template, *a, **kw):
        try:
            return _orig_formatter_init(self_fmt, template, *a, **kw)
        except Exception:
            # Skip broken templates — we use raw completion API
            self_fmt.template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            self_fmt.add_generation_prompt = kw.get("add_generation_prompt", True)
            from jinja2 import Environment
            self_fmt._template = Environment().from_string(self_fmt.template)

    lcf.Jinja2ChatFormatter.__init__ = _safe_formatter_init
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
    finally:
        lcf.Jinja2ChatFormatter.__init__ = _orig_formatter_init

    return llm


def _strip_thinking_tags(text: str) -> str:
    """Remove SmolLM3's <think>...</think>\n\n<thought>...</thought>\n\n<answer>...</answer> reasoning markers.

    SmolLM3 /no_think mode still outputs:
      思考\n\n...
解决方案\n<code>final_answer(...)</code>\n    We strip everything before the first <code> tag, and also remove the
    trailing </answer>\n markers.
    """
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # Remove 思考\n\n解决方案\n preamble before <code>
    # Keep everything from the first <code> tag onward
    code_match = re.search(r'<code>', text)
    if code_match:
        text = text[code_match.start():]
    # Clean trailing markers
    text = text.replace('</answer>\n', '').replace('</answer>', '')
    text = text.replace('</thought>', '').replace('</thought>\n', '')
    return text.strip()


class SmolLMModel(Model):
    """smolagents Model backed by local SmolLM3-3B via llama-cpp-python.

    Zero external tokens. Runs entirely on local CPU/GPU.
    Outputs <code>tool()</code> blocks natively for smolagents CodeAgent.

    Uses raw completion API to bypass SmolLM3's broken jinja2 chat template.
    Strips 思考/解决方案 reasoning markers from /no_think mode.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        model_id: str = "SmolLM3-3B-Q4_K_M",
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.model_path = model_path or str(DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILE)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads or os.cpu_count() or 4
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    def _load_model(self):
        """Lazy-load the GGUF model, bypassing SmolLM3's broken jinja2 template."""
        if self._llm is not None:
            return

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"SmolLM3 GGUF not found at {self.model_path}. "
                f"Download it with:\n"
                f"  mkdir -p {DEFAULT_MODEL_DIR}\n"
                f"  curl -L -o {DEFAULT_MODEL_DIR}/{DEFAULT_MODEL_FILE} "
                f"https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/resolve/main/SmolLM3-Q4_K_M.gguf"
            )

        logger.info(f"Loading SmolLM3 from {self.model_path} (n_ctx={self.n_ctx}, n_threads={self.n_threads})")
        self._llm = _load_llm_bypassing_chat_template(
            self.model_path, self.n_ctx, self.n_gpu_layers, self.n_threads
        )
        logger.info("SmolLM3 model loaded successfully")

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: Optional[list[str]] = None,
        response_format: Optional[dict] = None,
        tools_to_call_from: Optional[list] = None,
        **kwargs,
    ) -> ChatMessage:
        """Generate a response using local SmolLM3 inference.

        Constructs a chatml prompt manually and uses raw completion API.
        Strips SmolLM3's thinking/reasoning markers before returning.
        """
        self._load_model()

        # Build system prompt with tool descriptions
        system_text = SYSTEM_PROMPT

        if tools_to_call_from:
            tool_descs = []
            for tool in tools_to_call_from:
                if hasattr(tool, "name") and hasattr(tool, "description"):
                    sig = ""
                    if hasattr(tool, "inputs"):
                        params = []
                        for k, v in tool.inputs.items():
                            ptype = v.get("type", "string") if isinstance(v, dict) else "any"
                            pdesc = v.get("description", "") if isinstance(v, dict) else ""
                            params.append(f"{k}: {ptype}" + (f" — {pdesc}" if pdesc else ""))
                        if params:
                            sig = f"({', '.join(params)})"
                    tool_descs.append(f"- {tool.name}{sig}: {tool.description}")
            if tool_descs:
                system_text += "\n" + "\n".join(tool_descs)

        system_text += "\n<|im_end|>\n"

        prompt_parts = [system_text]

        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    str(item) if isinstance(item, str) else str(item)
                    for item in content
                )
            content = str(content) if content else ""
            if not content:
                continue

            role_tag = "user" if msg.role != MessageRole.ASSISTANT else "assistant"
            if msg.role == MessageRole.SYSTEM:
                role_tag = "system"

            if msg.role == MessageRole.ASSISTANT:
                truncated = content[:500] if len(content) > 500 else content
                prompt_parts.append(f"<|im_start|>{role_tag}\n{truncated}<|im_end|>")
            else:
                prompt_parts.append(f"<|im_start|>{role_tag}\n{content}<|im_end|>")

        prompt_parts.append("<|im_start|>assistant\n")
        full_prompt = "\n".join(prompt_parts)

        # Truncate to fit context window
        max_prompt_chars = (self.n_ctx - self.max_tokens) * 4
        if len(full_prompt) > max_prompt_chars:
            full_prompt = full_prompt[:max_prompt_chars] + "\n<|im_end|>\n<|im_start|>assistant\n"

        # Stop sequences — include SmolLM3 reasoning markers to cut off early
        stops = list(stop_sequences) if stop_sequences else []
        for s in ["<|im_end|>", "<|im_start|>"]:
            if s not in stops:
                stops.append(s)

        try:
            response = self._llm(
                full_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=stops,
                echo=False,
            )

            generated = response["choices"][0]["text"].strip()
            prompt_tokens = response.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = response.get("usage", {}).get("completion_tokens", 0)

        except Exception as e:
            logger.error(f"SmolLM3 inference error: {e}")
            generated = '<code>final_answer("Error: local model inference failed.")</code>'
            prompt_tokens = 0
            completion_tokens = 0

        # Strip SmolLM3 thinking markers
        generated = _strip_thinking_tags(generated)

        if not generated or not generated.strip():
            generated = '<code>final_answer("I could not generate a response.")</code>'

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=generated,
            token_usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )
