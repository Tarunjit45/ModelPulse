from typing import Literal

Tone = Literal["instructive", "neutral", "speculative"]

def classify_tone(text: str) -> Tone:
    """
    Classifies the tone of the given text based on simple keyword rules.
    """
    text_lower = text.lower()

    instructive_keywords = ["how to", "explain", "steps", "guide", "procedure", "instructions", "tutorial", "first", "next", "finally"]
    speculative_keywords = ["could", "might", "perhaps", "maybe", "imagine", "possibly", "hypothetically", "if", "would", "potential"]

    if any(keyword in text_lower for keyword in instructive_keywords):
        return "instructive"
    if any(keyword in text_lower for keyword in speculative_keywords):
        return "speculative"
    
    return "neutral"

def calculate_tone_drift_penalty(baseline_tone: Tone, current_tone: Tone) -> float:
    """
    Calculates a penalty if the tone has drifted.
    Returns 1.0 if tone changes, 0.0 otherwise.
    """
    return 1.0 if baseline_tone != current_tone else 0.0

if __name__ == "__main__":
    print("--- Tone Drift Testing ---")

    # Test instructive
    text1 = "Here are the steps to follow: First, open the application. Next, navigate to settings."
    print(f"'{text1}' -> Tone: {classify_tone(text1)}") # Expected: instructive

    # Test speculative
    text2 = "It could possibly be raining later. Perhaps we should bring an umbrella."
    print(f"'{text2}' -> Tone: {classify_tone(text2)}") # Expected: speculative

    # Test neutral
    text3 = "The quick brown fox jumps over the lazy dog."
    print(f"'{text3}' -> Tone: {classify_tone(text3)}") # Expected: neutral

    # Test preference for instructive if both present (order matters)
    text4 = "Explain how to reset your password, but it might not work immediately."
    print(f"'{text4}' -> Tone: {classify_tone(text4)}") # Expected: instructive (because 'how to' is checked first)

    # Test tone drift penalty
    print("\n--- Tone Drift Penalty Testing ---")
    print(f"Instructive vs Instructive: {calculate_tone_drift_penalty('instructive', 'instructive')}") # Expected: 0.0
    print(f"Instructive vs Speculative: {calculate_tone_drift_penalty('instructive', 'speculative')}") # Expected: 1.0
    print(f"Neutral vs Instructive: {calculate_tone_drift_penalty('neutral', 'instructive')}") # Expected: 1.0
