"""Multi-adapter inference engine with dynamic adapter loading."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from src.config import (
    ADAPTER_REGISTRY,
    INFERENCE_CONFIG,
    MODEL_NAME,
)
from src.model import load_base_model, load_tokenizer


class MultiAdapterModel:
    """Manages base model with multiple LoRA adapters loaded simultaneously.

    The general adapter is always active. Task adapters can be loaded and
    switched at runtime without reloading the base model.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer: AutoTokenizer = load_tokenizer(model_name)
        self.base_model = load_base_model(model_name)
        self.model: PeftModel | None = None
        self.loaded_adapters: set[str] = set()
        self.active_adapters: list[str] = []

    def load_adapter(self, adapter_name: str, adapter_path: str | Path) -> None:
        """Load a named adapter onto the base model."""
        if adapter_name in self.loaded_adapters:
            return

        if self.model is None:
            self.model = PeftModel.from_pretrained(
                self.base_model, str(adapter_path), adapter_name=adapter_name
            )
        else:
            self.model.load_adapter(str(adapter_path), adapter_name=adapter_name)

        self.loaded_adapters.add(adapter_name)

    def load_from_registry(self, adapter_name: str) -> bool:
        """Load an adapter by looking it up in the config registry.

        Returns True if loaded successfully, False if not found or not trained.
        """
        entry = ADAPTER_REGISTRY.get(adapter_name)
        if entry is None:
            print(f"Adapter '{adapter_name}' not found in registry")
            return False
        if not entry["trained"]:
            print(f"Adapter '{adapter_name}' is not trained yet (placeholder)")
            return False

        self.load_adapter(adapter_name, entry["path"])
        return True

    def set_active(self, adapter_name: str) -> None:
        """Set which adapter is active (applied during forward pass).

        Only one adapter can be active at a time in PEFT 0.19+.
        """
        if self.model is None:
            raise RuntimeError("No adapters loaded. Call load_adapter first.")

        if adapter_name not in self.loaded_adapters:
            raise ValueError(f"Adapter not loaded: {adapter_name}")

        self.model.set_adapter(adapter_name)
        self.active_adapters = [adapter_name]

    def activate_task(self, task_name: str) -> bool:
        """Convenience: load and activate a task adapter.

        Returns True if the task adapter was successfully activated.
        """
        if task_name not in self.loaded_adapters and not self.load_from_registry(task_name):
            return False

        self.set_active(task_name)
        return True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = INFERENCE_CONFIG["max_new_tokens"],
        temperature: float = INFERENCE_CONFIG["temperature"],
        top_p: float = INFERENCE_CONFIG["top_p"],
    ) -> str:
        """Generate a response using the currently active adapters."""
        if self.model is None:
            raise RuntimeError("No model loaded.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(self.tokenizer.decode(inputs["input_ids"][0])) :].strip()
        return response

    def chat(
        self,
        message: str,
        max_new_tokens: int = INFERENCE_CONFIG["max_new_tokens"],
        temperature: float = INFERENCE_CONFIG["temperature"],
        top_p: float = INFERENCE_CONFIG["top_p"],
    ) -> str:
        """Format a user message as an instruct prompt and generate."""
        prompt = f"Instruct: {message}\nOutput:"
        return self.generate(prompt, max_new_tokens, temperature, top_p)

    def info(self) -> dict:
        """Return current state of the model and adapters."""
        return {
            "base_model": MODEL_NAME,
            "loaded_adapters": sorted(self.loaded_adapters),
            "active_adapters": self.active_adapters,
            "registry": {
                name: {
                    "trained": entry["trained"],
                    "description": entry["description"],
                }
                for name, entry in ADAPTER_REGISTRY.items()
            },
        }
