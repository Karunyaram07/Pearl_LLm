# import gradio as gr

# from prompt_teacher.callbacks import (explain_improvement, explain_metaprompt,
#                                       robustly_improve_prompt, update_widgets)
# from prompt_teacher.messages import *
# from prompt_teacher.metaprompts import metaprompts

# with gr.Blocks(title="Prompt Teacher", theme=gr.themes.Soft()) as gradio_app:
#     gr.Markdown("### 🤖 Prompt Teacher 📝✨")
#     with gr.Accordion("ℹ️ Info: Code 📜 and Documentation 📚", open=True):
#         gr.Markdown(
#             "Can be found at: [Github: pwenker/prompt_teacher](https://github.com/pwenker/prompt_teacher) 📄✨"
#         )
#     with gr.Row():
#         with gr.Column(scale=2):
#             prompt_teacher = gr.Chatbot(
#                 height=580,
#                 label="Prompt Teacher",
#                 show_copy_button=True,
#                 value=[[inital_usr_text, initial_bot_text]],
#                 avatar_images=("thinking.svg", "robot.svg"),
#             )
#             prompt = gr.Textbox(
#                 label="Prompt",
#                 interactive=True,
#                 placeholder="Type in your prompt",
#                 value="How to write a good prompt?",
#                 show_copy_button=True,
#             )
#             with gr.Row():
#                 explain_btn = gr.Button(
#                     "Explain improvement 💡",
#                     variant="primary",
#                     visible=False,
#                 )
#                 replace_btn = gr.Button(
#                     "Accept improvement 👍",
#                     variant="primary",
#                     visible=False,
#                 )
#             with gr.Row():
#                 improve_btn = gr.Button("✨Improve prompt", variant="primary")
#         with gr.Column(scale=1):
#             model_name = gr.Dropdown(
#                 label="Large Language Model",
#                 info="Select Large Language Model",
#                 choices=[
#                     ("gpt-4o", "gpt-4o"),
#                     ("gpt-4-turbo", "gpt-4-turbo"),
#                     ("claude-3-opus", "claude-3-opus-20240229"),
#                 ],
#                 value="gpt-4o",
#             )
#             api_key = gr.Textbox(
#                 placeholder="Paste in your API key (sk-...)",
#                 label="OpenAI/Anthropic API Key",
#                 info="Paste in your API key",
#                 lines=1,
#                 type="password",
#             )
#             metaprompt = gr.Radio(
#                 label="Improvements",
#                 info="Select how the prompt should be improved",
#                 value="Comprehensive prompt refinement",
#                 choices=[mp.name.replace("_", " ").capitalize() for mp in metaprompts],
#             )
#             feedback = gr.Textbox(
#                 label="Feedback",
#                 info="Write your own feedback to be used to improve the prompt",
#                 visible=False,
#             )

#     improved_prompt = gr.Textbox(label="Improved Prompt", visible=False)
#     examples = gr.Examples(
#         examples=[[mp.name, mp.example_prompt] for mp in metaprompts],
#         examples_per_page=100,
#         inputs=[metaprompt, prompt],
#     )

#     metaprompt.change(
#         fn=update_widgets,
#         inputs=[metaprompt, feedback],
#         outputs=[improve_btn, feedback],
#     ).success(
#         lambda: [gr.Button(visible=False), gr.Button(visible=False)],
#         None,
#         [replace_btn, explain_btn],
#     ).success(
#         fn=explain_metaprompt,
#         inputs=[prompt_teacher, metaprompt],
#         outputs=[prompt_teacher],
#     )
#     improve_btn.click(
#         fn=robustly_improve_prompt,
#         inputs=[
#             model_name,
#             api_key,
#             prompt,
#             metaprompt,
#             feedback,
#             prompt_teacher,
#         ],
#         outputs=[improved_prompt, prompt_teacher],
#     ).success(
#         lambda: [gr.Button(visible=True), gr.Button(visible=True)],
#         None,
#         [replace_btn, explain_btn],
#     )

#     explain_btn.click(lambda: gr.Button(visible=False), None, explain_btn).success(
#         explain_improvement,
#         [
#             model_name,
#             api_key,
#             prompt,
#             metaprompt,
#             improved_prompt,
#             prompt_teacher,
#         ],
#         prompt_teacher,
#     )

#     replace_btn.click(lambda x: x, improved_prompt, prompt).success(
#         lambda: [gr.Button(visible=False), gr.Button(visible=False)],
#         None,
#         [replace_btn, explain_btn],
#     )

# if __name__ == "__main__":
#     gradio_app.queue(default_concurrency_limit=10).launch(favicon_path="robot.svg")



import gradio as gr
from transformers import pipeline
from dotenv import load_dotenv
import os

# 🔹 Load Hugging Face token from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")

# 🔹 Initialize Hugging Face model pipeline (Free Inference API)
generator = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-beta",
    token=hf_token
)

# ------------------------------------------------------------------------
# 🔹 Define simple local replacements for improvement and explanation
# ------------------------------------------------------------------------
def robustly_improve_prompt(model_name, api_key, prompt, metaprompt, feedback, prompt_teacher):
    """
    Replace OpenAI/Anthropic API call with Hugging Face free model inference.
    """
    # Instruction for model
    improvement_instruction = (
        f"Improve the following prompt according to '{metaprompt}'. "
        f"Provide a clearer and more effective version. "
        f"Feedback: {feedback}\n\nPrompt:\n{prompt}"
    )

    # Generate improved prompt using Hugging Face model
    result = generator(improvement_instruction, max_new_tokens=200, temperature=0.7)
    improved_prompt = result[0]["generated_text"]

    # Append response to chat
    messages = prompt_teacher + [
        ["User", prompt],
        ["AI", improved_prompt]
    ]

    return improved_prompt, messages


def explain_improvement(model_name, api_key, prompt, metaprompt, improved_prompt, prompt_teacher):
    """
    Explain how the improved prompt is better using the Hugging Face model.
    """
    explanation_instruction = (
        f"Explain how this improved prompt is better.\n\n"
        f"Original Prompt:\n{prompt}\n\nImproved Prompt:\n{improved_prompt}"
    )

    result = generator(explanation_instruction, max_new_tokens=150, temperature=0.7)
    explanation = result[0]["generated_text"]

    messages = prompt_teacher + [
        ["User", "Explain how this improvement helps."],
        ["AI", explanation]
    ]
    return messages


# ------------------------------------------------------------------------
# 🔹 Simplified UI setup (rest unchanged)
# ------------------------------------------------------------------------
from prompt_teacher.messages import *
from prompt_teacher.metaprompts import metaprompts
from prompt_teacher.callbacks import update_widgets, explain_metaprompt

with gr.Blocks(title="Prompt Teacher", theme=gr.themes.Soft()) as gradio_app:
    gr.Markdown("### 🤖 Prompt Teacher (Hugging Face Edition) 📝✨")
    with gr.Accordion("ℹ️ Info: Modified to use free Hugging Face model", open=True):
        gr.Markdown(
            "This version uses the **Mistral-7B-Instruct** model from Hugging Face. "
            "No OpenAI or Anthropic API keys are needed."
        )

    with gr.Row():
        with gr.Column(scale=2):
            prompt_teacher = gr.Chatbot(
                height=580,
                label="Prompt Teacher",
                show_copy_button=True,
                value=[[inital_usr_text, initial_bot_text]],
                avatar_images=("thinking.svg", "robot.svg"),
            )
            prompt = gr.Textbox(
                label="Prompt",
                interactive=True,
                placeholder="Type your prompt here...",
                value="How to write a good prompt?",
                show_copy_button=True,
            )
            with gr.Row():
                explain_btn = gr.Button(
                    "Explain improvement 💡",
                    variant="primary",
                    visible=False,
                )
                replace_btn = gr.Button(
                    "Accept improvement 👍",
                    variant="primary",
                    visible=False,
                )
            with gr.Row():
                improve_btn = gr.Button("✨Improve prompt", variant="primary")

        with gr.Column(scale=1):
            model_name = gr.Dropdown(
                label="Model (Hugging Face only)",
                info="Now uses Mistral-7B-Instruct from Hugging Face",
                choices=["mistralai/Mistral-7B-Instruct"],
                value="mistralai/Mistral-7B-Instruct",
            )
            api_key = gr.Textbox(
                placeholder="No API key needed for Hugging Face token",
                label="Hugging Face Token (.env)",
                info="Already loaded from .env file",
                lines=1,
                type="password",
                value="Loaded ✔️",
                interactive=False
            )
            metaprompt = gr.Radio(
                label="Improvements",
                info="Select how the prompt should be improved",
                value="Comprehensive prompt refinement",
                choices=[mp.name.replace("_", " ").capitalize() for mp in metaprompts],
            )
            feedback = gr.Textbox(
                label="Feedback",
                info="Write your feedback to guide improvement",
                visible=False,
            )

    improved_prompt = gr.Textbox(label="Improved Prompt", visible=False)
    examples = gr.Examples(
        examples=[[mp.name, mp.example_prompt] for mp in metaprompts],
        examples_per_page=100,
        inputs=[metaprompt, prompt],
    )

    # Flow connections
    metaprompt.change(
        fn=update_widgets,
        inputs=[metaprompt, feedback],
        outputs=[improve_btn, feedback],
    ).success(
        lambda: [gr.Button(visible=False), gr.Button(visible=False)],
        None,
        [replace_btn, explain_btn],
    ).success(
        fn=explain_metaprompt,
        inputs=[prompt_teacher, metaprompt],
        outputs=[prompt_teacher],
    )

    improve_btn.click(
        fn=robustly_improve_prompt,
        inputs=[model_name, api_key, prompt, metaprompt, feedback, prompt_teacher],
        outputs=[improved_prompt, prompt_teacher],
    ).success(
        lambda: [gr.Button(visible=True), gr.Button(visible=True)],
        None,
        [replace_btn, explain_btn],
    )

    explain_btn.click(lambda: gr.Button(visible=False), None, explain_btn).success(
        explain_improvement,
        [model_name, api_key, prompt, metaprompt, improved_prompt, prompt_teacher],
        prompt_teacher,
    )

    replace_btn.click(lambda x: x, improved_prompt, prompt).success(
        lambda: [gr.Button(visible=False), gr.Button(visible=False)],
        None,
        [replace_btn, explain_btn],
    )

# ------------------------------------------------------------------------
# 🔹 Launch Gradio App
# ------------------------------------------------------------------------
if __name__ == "__main__":
    gradio_app.queue(default_concurrency_limit=3).launch(favicon_path="robot.svg")
