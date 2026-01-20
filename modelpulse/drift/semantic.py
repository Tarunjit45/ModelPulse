from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Optional

# Global variable to store the model to avoid reloading for each call
_semantic_model: Optional[SentenceTransformer] = None

def _load_semantic_model() -> SentenceTransformer:
    """
    Loads the sentence transformer model. Caches it globally.
    """
    global _semantic_model
    if _semantic_model is None:
        print("Loading sentence-transformer model ('all-MiniLM-L6-v2'). This may take a moment...")
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence-transformer model loaded.")
    return _semantic_model

def get_embedding(text: str) -> List[float]:
    """
    Generates a sentence embedding for the given text.
    """
    model = _load_semantic_model()
    embedding = model.encode(text)
    return embedding.tolist()

def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculates the cosine similarity between two embeddings.
    """
    # cosine_similarity expects 2D arrays (e.g., [[embedding1]], [[embedding2]])
    similarity = cosine_similarity(np.array([embedding1]), np.array([embedding2]))[0][0]
    return float(similarity)

def calculate_semantic_drift_penalty(baseline_similarity: float, current_similarity: float, threshold: float = 0.85) -> float:
    """
    Calculates a penalty based on semantic drift.
    Drift if current_similarity < threshold.
    Penalty is inversely proportional to the similarity below the threshold.
    Returns a penalty between 0 and 1.
    """
    if current_similarity >= threshold:
        return 0.0
    
    # If similarity is below threshold, calculate penalty.
    # The penalty increases as similarity drops further below the threshold.
    # Max penalty (1.0) if similarity is 0 or less.
    # A linear scale from threshold to 0 similarity.
    penalty = (threshold - current_similarity) / threshold
    return min(1.0, max(0.0, penalty))


if __name__ == "__main__":
    print("--- Semantic Drift Testing ---")

    # Ensure the model is loaded for testing
    model = _load_semantic_model()

    text1 = "The cat sat on the mat."
    text2 = "A feline rested on the rug."
    text3 = "The dog barked loudly."
    text4 = "The cat sat on the mat, and it was a fluffy cat." # More words, but semantically similar

    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    emb3 = get_embedding(text3)
    emb4 = get_embedding(text4)

    # High similarity
    sim_high = calculate_cosine_similarity(emb1, emb2)
    print(f"Similarity ('{text1}' vs '{text2}'): {sim_high:.4f}") # Expected: High

    # Low similarity
    sim_low = calculate_cosine_similarity(emb1, emb3)
    print(f"Similarity ('{text1}' vs '{text3}'): {sim_low:.4f}") # Expected: Low

    # Semantic similarity despite length difference
    sim_len = calculate_cosine_similarity(emb1, emb4)
    print(f"Similarity ('{text1}' vs '{text4}'): {sim_len:.4f}") # Expected: High

    print("\n--- Semantic Drift Penalty Testing (Threshold = 0.85) ---")

    # Test 1: No drift (similarity above threshold)
    penalty1 = calculate_semantic_drift_penalty(0.9, 0.95, threshold=0.85)
    print(f"Sim 0.95 (baseline ignored for penalty, current {0.95}) -> Penalty: {penalty1}") # Expected: 0.0

    # Test 2: Slight drift (similarity just below threshold)
    penalty2 = calculate_semantic_drift_penalty(0.9, 0.80, threshold=0.85)
    # (0.85 - 0.80) / 0.85 = 0.05 / 0.85 = ~0.058
    print(f"Sim 0.80 -> Penalty: {penalty2:.4f}") # Expected: ~0.058

    # Test 3: Moderate drift
    penalty3 = calculate_semantic_drift_penalty(0.9, 0.60, threshold=0.85)
    # (0.85 - 0.60) / 0.85 = 0.25 / 0.85 = ~0.294
    print(f"Sim 0.60 -> Penalty: {penalty3:.4f}") # Expected: ~0.294

    # Test 4: Significant drift (similarity near 0)
    penalty4 = calculate_semantic_drift_penalty(0.9, 0.10, threshold=0.85)
    # (0.85 - 0.10) / 0.85 = 0.75 / 0.85 = ~0.882
    print(f"Sim 0.10 -> Penalty: {penalty4:.4f}") # Expected: ~0.882

    # Test 5: Similarity 0
    penalty5 = calculate_semantic_drift_penalty(0.9, 0.0, threshold=0.85)
    # (0.85 - 0.0) / 0.85 = 1.0
    print(f"Sim 0.0 -> Penalty: {penalty5:.4f}") # Expected: 1.0
