"""Manual adapter switching with RAG integration.

When switching tasks:
1. The last model output is indexed into the RAG vector store.
2. The new task adapter is loaded (general adapter stays active).
3. Relevant context is retrieved from RAG for the next prompt.
"""

from __future__ import annotations

from src.config import ADAPTER_REGISTRY
from src.inference import MultiAdapterModel
from src.rag import RAGPipeline


class SwitchManager:
    """Orchestrates adapter switches with RAG context handoff."""

    def __init__(self, model: MultiAdapterModel, rag: RAGPipeline | None = None):
        self.model = model
        self.rag = rag or RAGPipeline()
        self.current_task: str | None = None
        self.last_output: str | None = None

    @property
    def available_tasks(self) -> list[str]:
        """List of task names that are trained and ready."""
        return [name for name, entry in ADAPTER_REGISTRY.items() if entry["trained"]]

    @property
    def all_tasks(self) -> list[str]:
        """List of all registered task names."""
        return list(ADAPTER_REGISTRY.keys())

    def start(self, task_name: str) -> str:
        """Initialize the first task adapter (no RAG indexing on first call).

        Returns a status message.
        """
        if not self.model.activate_task(task_name):
            return f"Failed to activate '{task_name}' — adapter not trained"

        self.current_task = task_name
        return (
            f"Started with task: {task_name}\n"
            f"Active adapters: {self.model.active_adapters}\n"
            f"RAG store size: {self.rag.store.size}"
        )

    def generate_and_switch(
        self,
        message: str,
        new_task: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        """Generate a response with the current adapter, then switch tasks.

        Returns a dict with the response, switch status, and any retrieved context.
        """
        # 1. Generate with current adapter
        response = self.model.chat(message, max_new_tokens, temperature, top_p)
        self.last_output = response

        # 2. Index the output into RAG before switching
        if self.current_task:
            self.rag.index_output(
                text=response,
                task_name=self.current_task,
                adapter_name=self.current_task,
                extra={"user_message": message},
            )

        # 3. Switch to the new task
        switch_ok = self.model.activate_task(new_task)
        old_task = self.current_task
        self.current_task = new_task

        # 4. Retrieve context relevant to the next (unseen) input
        retrieved = self.rag.retrieve(message)
        enriched_query = self.rag.inject_context(message, retrieved)

        return {
            "response": response,
            "old_task": old_task,
            "new_task": new_task,
            "switch_ok": switch_ok,
            "rag_entries_indexed": self.rag.store.size,
            "context_retrieved": len(retrieved),
            "enriched_query": enriched_query,
        }

    def chat_with_context(
        self,
        message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response with RAG context injected into the prompt.

        Call this after a switch to use retrieved context automatically.
        """
        enriched = self.rag.query_with_context(message)
        return self.model.chat(enriched, max_new_tokens, temperature, top_p)

    def status(self) -> dict:
        """Current state of the switch manager."""
        return {
            "current_task": self.current_task,
            "loaded_adapters": sorted(self.model.loaded_adapters),
            "active_adapters": self.model.active_adapters,
            "rag_store_size": self.rag.store.size,
            "available_tasks": self.available_tasks,
        }
