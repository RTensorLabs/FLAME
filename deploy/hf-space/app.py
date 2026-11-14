"""Gradio demo for the FLAME 30M Rust model.

Each request runs `flame generate --cpu` on the bundled model. The model folder must contain
runs/<RUN>/config.toml, runs/<RUN>/best/model.safetensors and the tokenizer the config names.
"""
import os
import subprocess

import gradio as gr

MODEL_DIR = os.environ.get("FLAME_MODEL_DIR", "model")
FLAME_BIN = os.environ.get("FLAME_BIN", "flame")
RUN = os.environ.get("FLAME_RUN", "runs/rust-30m-dgx-20261102-stage2")
MAX_PROMPT_CHARS = 2000  # keeps prompt + output inside the model's 1024-token context
TIMEOUT_S = 240


def generate(prompt: str, max_tokens: float, temperature: float, top_k: float, seed: float) -> str:
    prompt = (prompt or "")[:MAX_PROMPT_CHARS]
    if not prompt.strip():
        return "Type a prompt first, for example: fn main() {"
    cmd = [
        FLAME_BIN, "generate", "--cpu", "--run", RUN,
        "--prompt", prompt,
        "--max-tokens", str(int(max_tokens)),
        "--temperature", str(float(temperature)),
        "--top-k", str(int(top_k)),
        "--seed", str(int(seed)),
    ]
    try:
        result = subprocess.run(cmd, cwd=MODEL_DIR, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return "Generation took too long. Try fewer tokens."
    if result.returncode != 0:
        return "Generation failed:\n" + result.stderr[-2000:]
    return result.stdout


DESCRIPTION = """
**FLAME 30M**: a 30-million-parameter language model trained from scratch in Rust
([Candle](https://github.com/huggingface/candle)) on Rust code, Rust docs, English text and
rustc error messages. It is a small *base* model: it continues text and does not follow
instructions. Start a piece of Rust code or docs and it will try to continue it.

It runs on a free 2-CPU machine, so 150 tokens takes about 20–40 seconds. Requests are handled one at a time.
"""

EXAMPLES = [
    ["fn main() {"],
    ["/// Returns the largest element of a slice.\npub fn largest<T: PartialOrd>(list: &[T]) -> &T {"],
    ["use std::collections::HashMap;\n\nfn word_count(text: &str) -> HashMap<String, usize> {"],
    ["error[E0382]: borrow of moved value: `s`"],
]

with gr.Blocks(title="FLAME 30M Rust model") as demo:
    gr.Markdown("# 🔥 FLAME 30M: Rust code model")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=8, placeholder="fn main() {")
            with gr.Accordion("Sampling settings", open=False):
                max_tokens = gr.Slider(16, 300, value=150, step=1, label="Max new tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_k = gr.Slider(1, 200, value=40, step=1, label="Top-k")
                seed = gr.Number(value=42, precision=0, label="Seed")
            button = gr.Button("Generate", variant="primary")
        output = gr.Code(label="Output (prompt + continuation)", language="markdown", lines=20)
    gr.Examples(EXAMPLES, inputs=[prompt])
    button.click(generate, [prompt, max_tokens, temperature, top_k, seed], output, concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
