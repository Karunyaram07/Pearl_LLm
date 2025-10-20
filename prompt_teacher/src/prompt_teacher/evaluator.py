from sentence_transformers import SentenceTransformer, util

# small and fast embedding model for CPU
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_evaluator = None

def get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = SentenceTransformer(EMBED_MODEL_NAME)
    return _evaluator

def similarity_score(expected: str, actual: str) -> float:
    model = get_evaluator()
    emb1 = model.encode(expected, convert_to_tensor=True)
    emb2 = model.encode(actual, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2).item()
    return float(sim)
