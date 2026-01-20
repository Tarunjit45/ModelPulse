def get_length(text: str) -> int:
    """
    Calculates the number of words in a given text.
    A more advanced implementation could count tokens or characters.
    """
    return len(text.split())

def calculate_length_drift_penalty(baseline_length: int, current_length: int, threshold: float = 0.3) -> float:
    """
    Calculates a penalty based on the percentage change in length.
    Drift if change > ±threshold (e.g., 30%).
    Penalty is proportional to the deviation from the threshold.
    Returns a penalty between 0 and 1.
    """
    if baseline_length == 0:
        return 0.0 if current_length == 0 else 1.0 # Max penalty if baseline is empty but current isn't

    percentage_change = abs((current_length - baseline_length) / baseline_length)

    if percentage_change > threshold:
        # Calculate how much the change exceeds the threshold
        exceeds_threshold_by = percentage_change - threshold
        # Simple linear penalty: 0 if at threshold, 1 if change is twice the threshold
        # (e.g., if threshold is 0.3, a change of 0.6 or more gives max penalty)
        penalty = min(1.0, exceeds_threshold_by / threshold)
        return penalty
    return 0.0

if __name__ == "__main__":
    print("--- Length Drift Testing ---")
    
    # Test 1: No change
    bl1 = 100
    cl1 = 100
    penalty1 = calculate_length_drift_penalty(bl1, cl1)
    print(f"Baseline: {bl1}, Current: {cl1}, Penalty: {penalty1}") # Expected: 0.0

    # Test 2: Within threshold (e.g., 20% change)
    bl2 = 100
    cl2 = 120
    penalty2 = calculate_length_drift_penalty(bl2, cl2)
    print(f"Baseline: {bl2}, Current: {cl2}, Penalty: {penalty2}") # Expected: 0.0

    # Test 3: Exceeds threshold slightly (e.g., 35% change, threshold 30%)
    bl3 = 100
    cl3 = 135
    penalty3 = calculate_length_drift_penalty(bl3, cl3)
    # Exceeds by 0.05. Penalty: 0.05 / 0.3 = ~0.166
    print(f"Baseline: {bl3}, Current: {cl3}, Penalty: {penalty3}") # Expected: ~0.166

    # Test 4: Significantly exceeds threshold (e.g., 60% change, threshold 30%)
    bl4 = 100
    cl4 = 160
    penalty4 = calculate_length_drift_penalty(bl4, cl4)
    # Exceeds by 0.3. Penalty: 0.3 / 0.3 = 1.0
    print(f"Baseline: {bl4}, Current: {cl4}, Penalty: {penalty4}") # Expected: 1.0

    # Test 5: Negative change, within threshold
    bl5 = 100
    cl5 = 80
    penalty5 = calculate_length_drift_penalty(bl5, cl5)
    print(f"Baseline: {bl5}, Current: {cl5}, Penalty: {penalty5}") # Expected: 0.0

    # Test 6: Negative change, exceeds threshold
    bl6 = 100
    cl6 = 60
    penalty6 = calculate_length_drift_penalty(bl6, cl6)
    print(f"Baseline: {bl6}, Current: {cl6}, Penalty: {penalty6}") # Expected: 1.0

    # Test 7: Zero baseline length
    bl7 = 0
    cl7 = 10
    penalty7 = calculate_length_drift_penalty(bl7, cl7)
    print(f"Baseline: {bl7}, Current: {cl7}, Penalty: {penalty7}") # Expected: 1.0

    bl8 = 0
    cl8 = 0
    penalty8 = calculate_length_drift_penalty(bl8, cl8)
    print(f"Baseline: {bl8}, Current: {cl8}, Penalty: {penalty8}") # Expected: 0.0
