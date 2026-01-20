import re
from typing import List, Set

def extract_keywords(text: str) -> List[str]:
    """
    Extracts potential keywords from text.
    For simplicity, this version extracts capitalized words that are not
    at the beginning of a sentence. This is a heuristic for proper nouns.
    A more advanced version would use POS tagging.
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    keywords = set()
    for sentence in sentences:
        # Find all words that start with an uppercase letter
        # and are not the very first word of the sentence.
        # This is a heuristic to catch proper nouns and important concepts.
        words = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        if words:
            # Exclude the first word of the sentence if it's capitalized
            # as it might just be the start of a sentence.
            first_word_match = re.match(r'^\s*\b[A-Z][a-z]+\b', sentence)
            if first_word_match and first_word_match.group(0).strip() == words[0]:
                words = words[1:] # Remove the first word if it was matched at sentence start
            
            for word in words:
                keywords.add(word.lower()) # Store in lowercase to avoid case sensitivity issues

    return sorted(list(keywords))

def calculate_keyword_drift_penalty(baseline_keywords: List[str], current_keywords: List[str]) -> float:
    """
    Calculates a penalty if important baseline keywords are missing from the current output.
    Returns a penalty between 0 and 1, based on the proportion of lost keywords.
    """
    baseline_set = set(baseline_keywords)
    current_set = set(current_keywords)

    if not baseline_set:
        return 0.0 # No penalty if there were no baseline keywords

    lost_keywords = baseline_set - current_set
    
    if not lost_keywords:
        return 0.0 # No penalty if no keywords were lost

    # Penalty is the proportion of lost keywords relative to baseline keywords
    penalty = len(lost_keywords) / len(baseline_set)
    return penalty

if __name__ == "__main__":
    print("--- Keyword Drift Testing ---")

    text1 = "The capital of France is Paris. Eiffel Tower is a landmark."
    print(f"Text: '{text1}' -> Keywords: {extract_keywords(text1)}") # Expected: ['eiffel', 'france', 'paris', 'tower'] (order may vary)

    text2 = "Apple Inc. makes iPhones. Steve Jobs founded the company."
    print(f"Text: '{text2}' -> Keywords: {extract_keywords(text2)}") # Expected: ['apple', 'inc', 'iphones', 'jobs', 'steve'] (order may vary)

    text3 = "This is a simple sentence. No specific keywords here."
    print(f"Text: '{text3}' -> Keywords: {extract_keywords(text3)}") # Expected: []

    print("\n--- Keyword Drift Penalty Testing ---")
    
    # Test 1: No lost keywords
    bk1 = ["apple", "banana", "cherry"]
    ck1 = ["apple", "banana", "cherry", "date"]
    penalty1 = calculate_keyword_drift_penalty(bk1, ck1)
    print(f"Baseline: {bk1}, Current: {ck1}, Penalty: {penalty1}") # Expected: 0.0

    # Test 2: Some lost keywords
    bk2 = ["apple", "banana", "cherry"]
    ck2 = ["apple", "cherry"]
    penalty2 = calculate_keyword_drift_penalty(bk2, ck2)
    # 1 lost / 3 baseline = 0.333...
    print(f"Baseline: {bk2}, Current: {ck2}, Penalty: {penalty2}") # Expected: ~0.333

    # Test 3: All keywords lost
    bk3 = ["apple", "banana", "cherry"]
    ck3 = ["grape"]
    penalty3 = calculate_keyword_drift_penalty(bk3, ck3)
    # 3 lost / 3 baseline = 1.0
    print(f"Baseline: {bk3}, Current: {ck3}, Penalty: {penalty3}") # Expected: 1.0

    # Test 4: Empty baseline
    bk4 = []
    ck4 = ["apple"]
    penalty4 = calculate_keyword_drift_penalty(bk4, ck4)
    print(f"Baseline: {bk4}, Current: {ck4}, Penalty: {penalty4}") # Expected: 0.0
