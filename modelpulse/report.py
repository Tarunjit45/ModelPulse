from typing import List, Dict, Any
from datetime import datetime

def generate_report_content(
    model_name: str,
    health_score: int,
    health_status: str,
    drift_details: Dict[str, Any],
    affected_prompts: List[str]
) -> str:
    """
    Generates a human-readable health report for ModelPulse.
    """
    report = f"ModelPulse Health Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += "----------------------------------------\n"
    report += f"Model: {model_name}\n"
    report += f"Health Score: {health_score}/100 ({health_status})\n\n"

    report += "Drift Detected:\n"
    if not drift_details:
        report += "- No significant drift detected.\n"
    else:
        if 'semantic_drift' in drift_details:
            report += f"- Semantic Drift: {drift_details['semantic_drift']['change_percent']:.0f}% (similarity: {drift_details['semantic_drift']['current_similarity']:.2f})\n"
        if 'tone_drift' in drift_details:
            report += f"- Tone Drift: {drift_details['tone_drift']['baseline_tone']} -> {drift_details['tone_drift']['current_tone']}\n"
        if 'keyword_loss' in drift_details and drift_details['keyword_loss']:
            report += f"- Keyword Loss: {', '.join(drift_details['keyword_loss'])}\n"
        if 'length_drift' in drift_details:
            report += f"- Length Drift: {drift_details['length_drift']['change_percent']:.0f}%\n"
    
    report += "\nAffected Prompts:\n"
    if not affected_prompts:
        report += "- None\n"
    else:
        for prompt_id in affected_prompts:
            report += f"- {prompt_id}\n"

    return report

if __name__ == "__main__":
    print("--- Report Generation Testing ---")

    # Example 1: Stable report
    report1 = generate_report_content(
        model_name="llama3.2:1b",
        health_score=95,
        health_status="Stable",
        drift_details={},
        affected_prompts=[]
    )
    print("--- Stable Report ---")
    print(report1)

    # Example 2: Warning report with some drift
    report2 = generate_report_content(
        model_name="llama3.2:1b",
        health_score=76,
        health_status="Warning",
        drift_details={
            "semantic_drift": {"change_percent": -18, "current_similarity": 0.67},
            "tone_drift": {"baseline_tone": "instructive", "current_tone": "speculative"},
            "keyword_loss": ["income"],
            "length_drift": {"change_percent": -15}
        },
        affected_prompts=["tax_explanation_v1"]
    )
    print("\n--- Warning Report ---")
    print(report2)

    # Example 3: Critical report with more drift
    report3 = generate_report_content(
        model_name="llama3.2:1b",
        health_score=65,
        health_status="Critical",
        drift_details={
            "semantic_drift": {"change_percent": -25, "current_similarity": 0.60},
            "tone_drift": {"baseline_tone": "instructive", "current_tone": "speculative"},
            "keyword_loss": ["account settings", "security"],
            "length_drift": {"change_percent": 40}
        },
        affected_prompts=["password_reset_guide_v1", "security_faq"]
    )
    print("\n--- Critical Report ---")
    print(report3)
