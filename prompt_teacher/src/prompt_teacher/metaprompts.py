import yaml
from pydantic import BaseModel, ValidationError
from pathlib import Path
import pkgutil

class Metaprompt(BaseModel):
    explanation: str
    example_prompt: str
    example_prompt_explanation: str
    name: str
    template: str

    def __str__(self):
        return f"{self.name}: {self.explanation}"

class MetapromptLibrary(BaseModel):
    Metaprompts: list[Metaprompt]

def read_and_validate(file_path: str | None = None):
    """
    Load metaprompts.yml from file_path or from package data.
    Returns MetapromptLibrary(Metaprompts=[]) on any error.
    """
    try:
        if file_path:
            p = Path(file_path)
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            else:
                raise FileNotFoundError
        else:
            # try to load package data (metaprompts.yml next to this file)
            data_bytes = pkgutil.get_data(__name__.split(".")[0], "metaprompts.yml")
            if not data_bytes:
                raise FileNotFoundError
            data = yaml.safe_load(data_bytes.decode("utf-8"))

        validated = MetapromptLibrary(**data)
        print(f"✅ Loaded {len(validated.Metaprompts)} metaprompts.")
        return validated
    except Exception as e:
        print(f"⚠ Could not load metaprompts: {e}. Using empty list.")
        return MetapromptLibrary(Metaprompts=[])

# global variables
metaprompts_library = read_and_validate()
metaprompts = metaprompts_library.Metaprompts
metaprompts_dict = {mp.name: mp for mp in metaprompts}
