import os
from dotenv import load_dotenv

load_dotenv()

# Model Configuration
DEFAULT_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3-8b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# App Settings
DEVICE = "cpu"
METAPROMPT_FILE = "./metaprompts.yml"
