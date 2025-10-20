import gradio as gr

def update_widgets(selected_metaprompt, feedback_text):
    """
    Controls visibility and behavior of widgets when a metaprompt is chosen.
    """
    if selected_metaprompt:
        return (
            gr.update(visible=True),   # improve button
            gr.update(visible=True)    # feedback box
        )
    else:
        return (
            gr.update(visible=False),
            gr.update(visible=False)
        )

def explain_metaprompt(prompt_teacher, selected_metaprompt):
    explanation_msg = (
        f"✅ You selected *{selected_metaprompt}*. "
        "Let's refine your prompt accordingly!"
    )
    new_history = prompt_teacher + [{"role": "assistant", "content": explanation_msg}]
    return new_history
