"""Gradio chat interface with multi-adapter switching and RAG context."""

import argparse

import gradio as gr

from src.config import ADAPTER_REGISTRY, INFERENCE_CONFIG
from src.inference import MultiAdapterModel
from src.rag import RAGPipeline
from src.switch import SwitchManager


def build_ui(switch_mgr: SwitchManager) -> gr.Blocks:
    """Build the Gradio UI with adapter switching and RAG controls."""
    all_tasks = switch_mgr.all_tasks

    with gr.Blocks(title="Robin LoRA") as demo:
        gr.Markdown("# Robin LoRA")
        gr.Markdown("Phi-2.7B + general adapter + task adapter (multi-adapter with RAG)")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=400)
                msg = gr.Textbox(label="Message", placeholder="Type a message...")
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    switch_btn = gr.Button("Switch Task", variant="secondary")

            with gr.Column(scale=1):
                task_dropdown = gr.Dropdown(
                    choices=all_tasks,
                    value="general",
                    label="Task adapter",
                    info="General is always active",
                )
                status_text = gr.Textbox(label="Status", lines=5, interactive=False)
                rag_count = gr.Number(label="RAG entries indexed", value=0, interactive=False)
                with gr.Accordion("Generation settings", open=False):
                    max_tokens = gr.Slider(64, 512, value=INFERENCE_CONFIG["max_new_tokens"], step=64, label="Max tokens")
                    temperature = gr.Slider(0.1, 2.0, value=INFERENCE_CONFIG["temperature"], step=0.1, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=INFERENCE_CONFIG["top_p"], step=0.05, label="Top-p")

        def user_message(user_msg, chat_history):
            """Append user message to chat and clear input."""
            chat_history.append({"role": "user", "content": user_msg})
            return "", chat_history

        def respond(chat_history, task, max_tok, temp, top):
            """Generate a response with the current adapter."""
            if not chat_history:
                return chat_history, switch_mgr.status()

            last_user_msg = chat_history[-1]["content"]
            response = switch_mgr.model.chat(last_user_msg, max_tok, temp, top)
            chat_history.append({"role": "assistant", "content": response})
            return chat_history, switch_mgr.status()

        def switch_task(chat_history, new_task, max_tok, temp, top):
            """Switch adapter: index last output, load new adapter, show context."""
            if not chat_history:
                gr.Warning("No conversation to index. Switching adapter only.")
                switch_mgr.model.activate_task(new_task)
                return chat_history, switch_mgr.status(), switch_mgr.rag.store.size

            # Find the last user-assistant pair
            last_user_msg = None
            last_response = None
            for msg in reversed(chat_history):
                if msg["role"] == "assistant" and last_response is None:
                    last_response = msg["content"]
                elif msg["role"] == "user" and last_user_msg is None:
                    last_user_msg = msg["content"]
                if last_user_msg and last_response:
                    break

            if not last_user_msg or not last_response:
                switch_mgr.model.activate_task(new_task)
                return chat_history, switch_mgr.status(), switch_mgr.rag.store.size

            # Index and switch
            result = switch_mgr.generate_and_switch(
                last_user_msg, new_task, max_tok, temp, top
            )

            # Add system message showing the switch
            if result["context_retrieved"] > 0:
                context_info = (
                    f"[Switched from {result['old_task']} -> {result['new_task']}. "
                    f"{result['context_retrieved']} context entries retrieved from RAG.]"
                )
            else:
                context_info = (
                    f"[Switched from {result['old_task']} -> {result['new_task']}. "
                    f"No prior context found in RAG.]"
                )

            chat_history.append({"role": "assistant", "content": context_info})
            return chat_history, switch_mgr.status(), switch_mgr.rag.store.size

        # Wire up events
        msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
            respond, [chatbot, task_dropdown, max_tokens, temperature, top_p], [chatbot, status_text]
        )
        send_btn.click(user_message, [msg, chatbot], [msg, chatbot]).then(
            respond, [chatbot, task_dropdown, max_tokens, temperature, top_p], [chatbot, status_text]
        )
        switch_btn.click(
            switch_task,
            [chatbot, task_dropdown, max_tokens, temperature, top_p],
            [chatbot, status_text, rag_count],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="general",
        choices=list(ADAPTER_REGISTRY.keys()),
        help="Initial task adapter to load",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    model = MultiAdapterModel()
    rag = RAGPipeline()
    switch_mgr = SwitchManager(model, rag)

    print(switch_mgr.start(args.task))
    print()
    for name, entry in ADAPTER_REGISTRY.items():
        status = "trained" if entry["trained"] else "NOT trained (placeholder)"
        print(f"  {name}: {entry['description']} [{status}]")

    demo = build_ui(switch_mgr)
    demo.launch(share=args.share)
