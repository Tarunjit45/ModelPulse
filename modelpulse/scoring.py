from typing import Dict, Literal

def calculate_health_score(
    semantic_penalty: float,
    tone_penalty: float,
    length_penalty: float,
    keyword_penalty: float
) -> tuple[int, Literal["Stable", "Warning", "Critical"]]:
    """
    Computes the overall health score based on individual drift penalties.

    Health Score = 100
    - (semantic_penalty * 40)
    - (tone_penalty * 20)
    - (length_penalty * 20)
    - (keyword_penalty * 20)

    Returns the score and its corresponding status (Stable, Warning, Critical).
    """
    score = 100.0 \
            - (semantic_penalty * 40) \
            - (tone_penalty * 20) \
            - (length_penalty * 20) \
            - (keyword_penalty * 20)
    
    score = max(0.0, min(100.0, score)) # Ensure score is between 0 and 100

    status: Literal["Stable", "Warning", "Critical"]
    if score >= 90:
        status = "Stable"
    elif score >= 70:
        status = "Warning"
    else:
        status = "Critical"
    
    return int(score), status

if __name__ == "__main__":
    print("--- Health Score Testing ---")

    # Test 1: No drift
    score1, status1 = calculate_health_score(0, 0, 0, 0)
    print(f"Penalties (0,0,0,0) -> Score: {score1}, Status: {status1}") # Expected: 100, Stable

    # Test 2: Semantic drift only (slight)
    score2, status2 = calculate_health_score(0.1, 0, 0, 0) # 40 * 0.1 = 4
    print(f"Penalties (0.1,0,0,0) -> Score: {score2}, Status: {status2}") # Expected: 96, Stable

    # Test 3: Semantic drift only (moderate)
    score3, status3 = calculate_health_score(0.5, 0, 0, 0) # 40 * 0.5 = 20
    print(f"Penalties (0.5,0,0,0) -> Score: {score3}, Status: {status3}") # Expected: 80, Warning

    # Test 4: Tone drift only
    score4, status4 = calculate_health_score(0, 1.0, 0, 0) # 20 * 1.0 = 20
    print(f"Penalties (0,1.0,0,0) -> Score: {score4}, Status: {status4}") # Expected: 80, Warning

    # Test 5: Multiple drifts leading to Warning
    score5, status5 = calculate_health_score(0.2, 1.0, 0.1, 0) # 40*0.2 + 20*1.0 + 20*0.1 = 8 + 20 + 2 = 30
    print(f"Penalties (0.2,1.0,0.1,0) -> Score: {score5}, Status: {status5}") # Expected: 70, Warning

    # Test 6: Multiple drifts leading to Critical
    score6, status6 = calculate_health_score(0.5, 1.0, 0.5, 0.5) # 40*0.5 + 20*1.0 + 20*0.5 + 20*0.5 = 20 + 20 + 10 + 10 = 60
    print(f"Penalties (0.5,1.0,0.5,0.5) -> Score: {score6}, Status: {status6}") # Expected: 40, Critical

    # Test 7: Max penalty
    score7, status7 = calculate_health_score(1.0, 1.0, 1.0, 1.0) # 40+20+20+20 = 100
    print(f"Penalties (1.0,1.0,1.0,1.0) -> Score: {score7}, Status: {status7}") # Expected: 0, Critical
