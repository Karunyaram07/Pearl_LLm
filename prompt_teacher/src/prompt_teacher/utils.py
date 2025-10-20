import os
import torch
from transformers import pipeline
from huggingface_hub import InferenceClient

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_generator(model_name: str | None = None, hf_token: str | None = None):
    """
    Builds a generator using either:
      - Hugging Face Inference API (if token provided)
      - Local transformers pipeline (default)
    """
    device = get_device()

    # --- If Hugging Face API key is given, use Inference API ---
    if hf_token and model_name:
        client = InferenceClient(token=hf_token)

        def infer(prompt, max_new_tokens=150, temperature=0.7):
            try:
                # ✅ New correct API call (no "inputs=")
                out = client.text_generation(
                    prompt,
                    model=model_name,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True
                )
                # handle different output formats
                if isinstance(out, list):
                    return out[0].get("generated_text", "")
                elif isinstance(out, dict):
                    return out.get("generated_text", "")
                return str(out)
            except Exception as e:
                return f"⚠ Error from HF API: {e}"

        return infer, f"InferenceAPI:{model_name}"

    # --- Otherwise use local Transformers pipeline ---
    if not model_name:
        model_name = "distilgpt2"

    generator = pipeline(
        "text-generation",
        model=model_name,
        device=-1 if device == "cpu" else 0
    )

    def gen(prompt, max_new_tokens=150, temperature=0.7):
        res = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        return res[0].get("generated_text", "")

    return gen, model_name
