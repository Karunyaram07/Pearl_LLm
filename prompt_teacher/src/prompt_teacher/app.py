import os
from dotenv import load_dotenv
import gradio as gr
from huggingface_hub import InferenceClient
from pathlib import Path

# Internal imports
from prompt_teacher.messages import inital_usr_text, initial_bot_text
from prompt_teacher.metaprompts import metaprompts
from prompt_teacher.callbacks import update_widgets, explain_metaprompt
from prompt_teacher.evaluator import similarity_score

# -----------------------------
# Environment setup
# -----------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY", None)

# Default open-access model (small & fast)
DEFAULT_MODEL = os.getenv("DEFAULT_HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

# Optional gated model (requires HF approval)
GATED_MODEL = "meta-llama/Llama-3.1-8B"

# -----------------------------
# Initialize Hugging Face client
# -----------------------------
def init_hf_client(model_name, token=None):
    try:
        client = InferenceClient(model=model_name, token=token)
        print(f"✅ Using Hugging Face model: {model_name}")
        return client
    except Exception as e:
        print(f"⚠ Error initializing model '{model_name}': {e}")
        return None

# Try gated model if token exists, else fallback to open-access
client = init_hf_client(GATED_MODEL, HF_TOKEN) if HF_TOKEN else None
if client is None:
    print("⚠ Falling back to open-access model.")
    client = init_hf_client(DEFAULT_MODEL)

# -----------------------------
# Core Functions
# -----------------------------
def robustly_improve_prompt(model_name, api_key, prompt_text, metaprompt_name, feedback, chat_history):
    mp_obj = next((mp for mp in metaprompts if mp.name.strip().lower() == (metaprompt_name or "").strip().lower()), None)
    if not mp_obj:
        err = f"⚠ Could not find metaprompt named '{metaprompt_name}'."
        chat_history.append({"role": "assistant", "content": err})
        return err, chat_history

    # Fill template
    template = mp_obj.template
    filled = template.replace("{prompt}", prompt_text or "").replace("{feedback}", feedback or "")

    # Refined instruction
    instruction = (
        f"{mp_obj.explanation}\n\n"
        f"Original Prompt:\n{prompt_text}\n\n"
        f"Your task: Rewrite the above prompt to make it clearer, more detailed, and actionable. "
        f"Provide only the improved prompt text, no extra commentary."
    )

    try:
        response = client.text_generation(prompt=instruction, max_new_tokens=200, temperature=0.7)
        improved = response
    except Exception as e:
        improved = f"⚠ Error from HF API: {e}"

    chat_history.append({"role": "user", "content": prompt_text})
    chat_history.append({"role": "assistant", "content": improved})
    return improved, chat_history


def explain_improvement(model_name, api_key, prompt_text, metaprompt_name, improved_prompt, chat_history):
    instruction = (
        f"Explain concisely why the following improved prompt is better.\n\n"
        f"Original Prompt:\n{prompt_text}\n\nImproved Prompt:\n{improved_prompt}\n\n"
        f"Use clear bullet points and do not add extra commentary."
    )

    try:
        response = client.text_generation(prompt=instruction, max_new_tokens=150, temperature=0.7)
        explanation = response
    except Exception as e:
        explanation = f"⚠ Error: {e}"

    chat_history.append({"role": "user", "content": "Explain how this improvement helps."})
    chat_history.append({"role": "assistant", "content": explanation})
    return chat_history


def evaluate_example(expected, actual):
    try:
        score = similarity_score(expected, actual)
    except Exception:
        score = 0.0
    return f"Similarity score: {score:.3f}"


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Prompt Teacher", theme=gr.themes.Soft()) as app:
    gr.Markdown("### 🤖 Prompt Teacher (Hugging Face Edition) 📝✨")

    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.Chatbot(
                label="Prompt Teacher",
                height=520,
                value=[{"role": "user", "content": inital_usr_text},
                       {"role": "assistant", "content": initial_bot_text}],
                elem_id="chat",
                show_copy_button=True,
                type="messages"
            )

            prompt = gr.Textbox(label="Prompt", placeholder="Type your prompt here...", value="How to write a good prompt?")
            with gr.Row():
                explain_btn = gr.Button("Explain improvement 💡", visible=False)
                replace_btn = gr.Button("Accept improvement 👍", visible=False)
            improve_btn = gr.Button("✨ Improve prompt", variant="primary")

        with gr.Column(scale=1):
            model_name = gr.Textbox(label="Model (Hugging Face)", value=DEFAULT_MODEL)
            api_key = gr.Textbox(label="Hugging Face Token (optional)", value="Loaded" if HF_TOKEN else "", interactive=False)
            metaprompt = gr.Radio(label="Improvements", choices=[mp.name for mp in metaprompts], value=metaprompts[0].name if metaprompts else None)
            feedback = gr.Textbox(label="Feedback (optional)", visible=False)

    improved_prompt = gr.Textbox(label="Improved Prompt", visible=False)
    example_expected = gr.Textbox(label="Expected answer (for evaluation)", placeholder="Provide expected answer to compute similarity", visible=True)
    example_actual = gr.Textbox(label="Actual answer (model output)", placeholder="Paste model output to evaluate", visible=True)
    eval_btn = gr.Button("Evaluate Example")

    # Interactions
    metaprompt.change(fn=update_widgets, inputs=[metaprompt, feedback], outputs=[improve_btn, feedback]).success(
        fn=explain_metaprompt, inputs=[chat, metaprompt], outputs=[chat]
    )

    improve_btn.click(
        fn=robustly_improve_prompt,
        inputs=[model_name, api_key, prompt, metaprompt, feedback, chat],
        outputs=[improved_prompt, chat]
    ).success(lambda: [gr.update(visible=True), gr.update(visible=True)], None, [replace_btn, explain_btn])

    explain_btn.click(
        fn=explain_improvement,
        inputs=[model_name, api_key, prompt, metaprompt, improved_prompt, chat],
        outputs=[chat]
    )

    replace_btn.click(fn=lambda x: x, inputs=improved_prompt, outputs=prompt).success(lambda: [gr.update(visible=False), gr.update(visible=False)], None, [replace_btn, explain_btn])

    eval_btn.click(fn=evaluate_example, inputs=[example_expected, example_actual], outputs=[chat])


# -----------------------------
# Launch
# -----------------------------
robot_icon = os.path.join(os.path.dirname(__file__), "robot.svg")
if __name__ == "__main__":
    app.queue().launch(favicon_path=robot_icon)
