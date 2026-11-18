"""
NEXUS NLP — Modular Language Model Inference Engine
Encapsulates Google FLAN-T5 Seq2Seq model lifecycle, tokenization, decoding strategies,
parameter validation, and performance profiling.

Author: Subham Gupta (ShadowFox ML Engineer Intern)
Week 3: Task 3 - Language Model Deployment & Intelligence Evaluation Lab
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union

import torch
import psutil
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizerFast

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("NexusModelEngine")


@dataclass
class GenerationResult:
    """Encapsulates output text, performance metrics, and generation metadata."""
    generated_text: str
    prompt: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_per_second: float
    decoding_strategy: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    memory_rss_mb: Optional[float] = None


class NexusModelEngine:
    """
    Production-grade inference engine for FLAN-T5-small on CPU architecture.
    Provides strict parameter validation, decoupled decoding strategies,
    high-precision millisecond latency profiling, and batch generation capabilities.
    """

    SUPPORTED_STRATEGIES = ["greedy", "beam_search", "sampling", "custom"]

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        cache_dir: Optional[str] = None,
        device: str = "cpu"
    ):
        self.model_name_or_path = model_name_or_path

        # Default local cache inside Task-3 directory
        if cache_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.cache_dir = os.path.join(base_dir, "model_cache")
        else:
            self.cache_dir = cache_dir

        self.device = torch.device(device)
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.model: Optional[PreTrainedModel] = None
        self.process = psutil.Process(os.getpid())

        # Load state metrics
        self.load_time_sec: float = 0.0
        self.model_memory_delta_mb: float = 0.0
        self.is_loaded: bool = False

        # Architectural details cache
        self.config_spec: Dict[str, Any] = {}

    def load(self) -> "NexusModelEngine":
        """
        Loads tokenizer and model from local cache into memory on the target device.
        Measures load time and RAM footprint delta.
        """
        if self.is_loaded:
            logger.info("Model is already loaded and initialized.")
            return self

        logger.info(f"Loading tokenizer and model: {self.model_name_or_path} (cache: {self.cache_dir})")
        mem_before_mb = self.process.memory_info().rss / (1024 * 1024)
        t0 = time.perf_counter()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()

            # Cache architecture details
            cfg = self.model.config
            self.config_spec = {
                "model_type": getattr(cfg, "model_type", "t5"),
                "d_model": cfg.d_model,
                "d_kv": cfg.d_kv,
                "d_ff": cfg.d_ff,
                "num_layers": cfg.num_layers,
                "num_decoder_layers": cfg.num_decoder_layers,
                "num_heads": cfg.num_heads,
                "relative_attention_num_buckets": cfg.relative_attention_num_buckets,
                "feed_forward_proj": getattr(cfg, "feed_forward_proj", "relu"),
                "vocab_size": cfg.vocab_size,
                "total_parameters": sum(p.numel() for p in self.model.parameters())
            }

        except Exception as e:
            logger.error(f"Failed to load model from {self.model_name_or_path}: {e}")
            raise RuntimeError(f"Error loading model or tokenizer: {e}") from e

        self.load_time_sec = time.perf_counter() - t0
        mem_after_mb = self.process.memory_info().rss / (1024 * 1024)
        self.model_memory_delta_mb = max(0.0, mem_after_mb - mem_before_mb)
        self.is_loaded = True

        logger.info(
            f"Successfully initialized {self.model_name_or_path} on {self.device} in "
            f"{self.load_time_sec:.2f}s (RAM delta: +{self.model_memory_delta_mb:.2f} MB)"
        )
        return self

    def validate_inputs(
        self,
        prompt: str,
        max_new_tokens: int,
        strategy: str,
        do_sample: bool,
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        num_beams: int,
        repetition_penalty: float
    ) -> None:
        """
        Validates prompt text and hyperparameter compatibility.
        Guarantees that sampling-only parameters are never erroneously combined
        with deterministic decoding strategies.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0 or max_new_tokens > 1024:
            raise ValueError(f"max_new_tokens must be an integer between 1 and 1024. Got: {max_new_tokens}")

        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy '{strategy}'. Supported: {self.SUPPORTED_STRATEGIES}")

        if not isinstance(num_beams, int) or num_beams < 1:
            raise ValueError(f"num_beams must be an integer >= 1. Got: {num_beams}")

        if repetition_penalty < 1.0 or repetition_penalty > 5.0:
            raise ValueError(f"repetition_penalty must be between 1.0 and 5.0. Got: {repetition_penalty}")

        # Deterministic vs Stochastic parameter orthogonality
        if not do_sample:
            if temperature is not None and temperature != 1.0:
                raise ValueError(
                    f"Temperature={temperature} specified with do_sample=False. "
                    "Temperature is only valid when sampling is enabled."
                )
            if top_k is not None and top_k != 50 and top_k > 0:
                raise ValueError(
                    f"top_k={top_k} specified with do_sample=False. "
                    "top_k is only valid when sampling is enabled."
                )
            if top_p is not None and top_p < 1.0:
                raise ValueError(
                    f"top_p={top_p} specified with do_sample=False. "
                    "top_p is only valid when sampling is enabled."
                )
        else:
            if temperature is not None and (temperature <= 0.0 or temperature > 3.0):
                raise ValueError(f"temperature must be between 0.01 and 3.0. Got: {temperature}")
            if top_k is not None and top_k < 0:
                raise ValueError(f"top_k must be >= 0. Got: {top_k}")
            if top_p is not None and (top_p <= 0.0 or top_p > 1.0):
                raise ValueError(f"top_p must be in range (0.0, 1.0]. Got: {top_p}")

    def generate(
        self,
        prompt: str,
        strategy: str = "greedy",
        max_new_tokens: int = 64,
        num_beams: int = 1,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> GenerationResult:
        """
        Executes text generation on the loaded Seq2Seq model.
        Configures generation arguments safely based on the chosen strategy.
        """
        if not self.is_loaded:
            self.load()

        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy '{strategy}'. Supported: {self.SUPPORTED_STRATEGIES}")

        # Strategy resolution & parameter defaulting
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }

        if strategy == "greedy":
            do_sample = False
            num_beams = 1
            temperature = None
            top_k = None
            top_p = None
            gen_kwargs["num_beams"] = 1
            gen_kwargs["do_sample"] = False

        elif strategy == "beam_search":
            do_sample = False
            if num_beams <= 1:
                num_beams = 4  # Default beam width for beam search
            temperature = None
            top_k = None
            top_p = None
            gen_kwargs["num_beams"] = num_beams
            gen_kwargs["do_sample"] = False
            gen_kwargs["early_stopping"] = kwargs.get("early_stopping", True)

        elif strategy == "sampling":
            do_sample = True
            gen_kwargs["do_sample"] = True
            gen_kwargs["num_beams"] = 1
            temp_val = temperature if temperature is not None else 0.7
            top_k_val = top_k if top_k is not None else 50
            top_p_val = top_p if top_p is not None else 0.9
            gen_kwargs["temperature"] = temp_val
            gen_kwargs["top_k"] = top_k_val
            gen_kwargs["top_p"] = top_p_val
            temperature, top_k, top_p = temp_val, top_k_val, top_p_val

        elif strategy == "custom":
            do_sample = kwargs.get("do_sample", False)
            gen_kwargs["do_sample"] = do_sample
            gen_kwargs["num_beams"] = num_beams
            if do_sample:
                if temperature is not None:
                    gen_kwargs["temperature"] = temperature
                if top_k is not None:
                    gen_kwargs["top_k"] = top_k
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Strict input validation
        self.validate_inputs(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            strategy=strategy,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty
        )

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_token_count = inputs["input_ids"].shape[1]

        # Timed inference execution
        t0 = time.perf_counter()
        with torch.no_grad():
            output_sequences = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        elapsed_sec = time.perf_counter() - t0
        latency_ms = elapsed_sec * 1000.0

        # Decode output (Seq2Seq generates decoder-only tokens starting with decoder_start_token_id)
        generated_tokens_count = output_sequences.shape[1]
        decoded_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True).strip()

        tokens_per_second = (generated_tokens_count / elapsed_sec) if elapsed_sec > 0 else 0.0
        current_rss_mb = self.process.memory_info().rss / (1024 * 1024)

        recorded_params = {
            "strategy": strategy,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty
        }

        return GenerationResult(
            generated_text=decoded_text,
            prompt=prompt,
            latency_ms=round(latency_ms, 2),
            input_tokens=input_token_count,
            output_tokens=generated_tokens_count,
            total_tokens=input_token_count + generated_tokens_count,
            tokens_per_second=round(tokens_per_second, 2),
            decoding_strategy=strategy,
            parameters=recorded_params,
            memory_rss_mb=round(current_rss_mb, 2)
        )

    def generate_batch(
        self,
        prompts: List[str],
        strategy: str = "greedy",
        max_new_tokens: int = 64,
        **kwargs
    ) -> List[GenerationResult]:
        """Runs sequential generation across a list of prompts."""
        results = []
        for p in prompts:
            res = self.generate(
                prompt=p,
                strategy=strategy,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            results.append(res)
        return results

    def get_specs(self) -> Dict[str, Any]:
        """Returns loaded model specifications and performance telemetry."""
        if not self.is_loaded:
            self.load()
        return {
            "model_name_or_path": self.model_name_or_path,
            "device": str(self.device),
            "load_time_sec": round(self.load_time_sec, 3),
            "memory_delta_mb": round(self.model_memory_delta_mb, 2),
            "config": self.config_spec
        }
