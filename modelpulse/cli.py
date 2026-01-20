import typer
from pathlib import Path
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from modelpulse.baseline import Baseline, ModelConfig, ExtractedMetadata
from modelpulse.runner import LLMRunner, MockLLMRunner
from modelpulse.drift.length import get_length, calculate_length_drift_penalty
from modelpulse.drift.tone import classify_tone, calculate_tone_drift_penalty
from modelpulse.drift.keywords import extract_keywords, calculate_keyword_drift_penalty
from modelpulse.drift.semantic import get_embedding, calculate_cosine_similarity, calculate_semantic_drift_penalty, _load_semantic_model
from modelpulse.scoring import calculate_health_score
from modelpulse.report import generate_report_content

app = typer.Typer()

BASE_DIR = Path(__file__).resolve().parent.parent # modelpulse/ 

# Ensure semantic model is loaded at startup if needed
# This can be made lazy if startup time is an issue and semantic analysis isn't always used
_ = _load_semantic_model()

@app.command()
def init():
    """
    Initializes the ModelPulse project by creating necessary directories.
    """
    typer.echo("Initializing ModelPulse project...")

    dirs_to_create = [
        BASE_DIR / "baselines",
        BASE_DIR / "prompts",
        BASE_DIR / "runs",
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Created directory: {d}")

    typer.echo("ModelPulse project initialized successfully.")

@app.command()
def baseline(
    prompts_dir: Path = typer.Argument(BASE_DIR / "prompts", help="Path to the directory containing prompt files."),
    model_name: str = typer.Option("llama3", "--model", "-m", help="Name of the LLM to use for generating baselines."),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Model temperature setting."),
    top_p: float = typer.Option(0.9, "--top-p", help="Model top-p setting for nucleus sampling."),
    use_mock_llm: bool = typer.Option(False, "--mock", "-M", help="Use a mock LLM runner instead of Ollama."),
):
    """
    Creates baseline outputs for critical prompts.
    Reads prompt files, calls the model, stores baseline outputs and extracted metadata.
    """
    typer.echo(f"Creating baselines from prompts in '{prompts_dir}'...")

    if not prompts_dir.exists() or not prompts_dir.is_dir():
        typer.echo(f"Error: Prompts directory '{prompts_dir}' not found or is not a directory.")
        raise typer.Exit(code=1)

    llm_runner = MockLLMRunner(model_name) if use_mock_llm else LLMRunner(model_name)
    model_config = ModelConfig(temperature=temperature, top_p=top_p)
    baselines_dir = BASE_DIR / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    prompt_files = list(prompts_dir.glob("*.txt"))
    if not prompt_files:
        typer.echo(f"No .txt prompt files found in '{prompts_dir}'.")
        raise typer.Exit(code=0)

    for prompt_file in prompt_files:
        prompt_id = prompt_file.stem
        typer.echo(f"\nProcessing prompt: {prompt_id}")
        
        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        typer.echo(f"  Prompt text: {prompt_text[:70]}...")

        # Generate baseline output
        typer.echo("  Generating model response...")
        baseline_output = llm_runner.generate_response(prompt_text, model_config.model_dump())
        typer.echo(f"  Model response: {baseline_output[:70]}...")

        # Extract metadata
        typer.echo("  Extracting metadata...")
        tone = classify_tone(baseline_output)
        length = get_length(baseline_output)
        keywords = extract_keywords(baseline_output)
        semantic_embedding = get_embedding(baseline_output)

        extracted_metadata = ExtractedMetadata(
            tone=tone,
            length=length,
            keywords=keywords,
            semantic_embedding=semantic_embedding
        )

        baseline_obj = Baseline(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            model_name=model_name,
            model_config=model_config,
            baseline_output=baseline_output,
            extracted_metadata=extracted_metadata,
            timestamp=datetime.now()
        )

        baseline_file_path = baselines_dir / f"{prompt_id}.json"
        baseline_obj.to_json_file(baseline_file_path)
        typer.echo(f"  Baseline saved to {baseline_file_path}")

    typer.echo("\nAll baselines created successfully.")

@app.command()
def check(
    model_name: str = typer.Option("llama3", "--model", "-m", help="Name of the LLM to use for re-running prompts."),
    use_mock_llm: bool = typer.Option(False, "--mock", "-M", help="Use a mock LLM runner instead of Ollama."),
):
    """
    Re-runs prompts against the model, compares outputs to baselines, and detects drift.
    Calculates a health score and generates a drift report.
    """
    typer.echo("Checking for model drift...")

    baselines_dir = BASE_DIR / "baselines"
    runs_dir = BASE_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    baseline_files = list(baselines_dir.glob("*.json"))
    if not baseline_files:
        typer.echo(f"No baseline files found in '{baselines_dir}'. Please create baselines first using 'modelpulse baseline'.")
        raise typer.Exit(code=1)
    
    llm_runner = MockLLMRunner(model_name) if use_mock_llm else LLMRunner(model_name)

    overall_drift_details: Dict[str, Any] = {}
    overall_affected_prompts: List[str] = []
    
    total_semantic_penalty = 0.0
    total_tone_penalty = 0.0
    total_length_penalty = 0.0
    total_keyword_penalty = 0.0
    num_prompts_checked = 0

    for baseline_file in baseline_files:
        num_prompts_checked += 1
        baseline_obj = Baseline.from_json_file(baseline_file)
        typer.echo(f"\nComparing prompt '{baseline_obj.prompt_id}'...")

        # Re-run prompt
        current_output = llm_runner.generate_response(
            baseline_obj.prompt_text, 
            baseline_obj.model_config.model_dump()
        )
        typer.echo(f"  Current model response: {current_output[:70]}...")

        # Extract current metadata
        current_tone = classify_tone(current_output)
        current_length = get_length(current_output)
        current_keywords = extract_keywords(current_output)
        current_semantic_embedding = get_embedding(current_output)

        prompt_drift_details: Dict[str, Any] = {}
        prompt_has_drift = False

        # --- Calculate Semantic Drift ---
        baseline_semantic_embedding = baseline_obj.extracted_metadata.semantic_embedding
        semantic_similarity = 0.0
        if baseline_semantic_embedding:
            semantic_similarity = calculate_cosine_similarity(baseline_semantic_embedding, current_semantic_embedding)
            semantic_penalty = calculate_semantic_drift_penalty(0.85, semantic_similarity) # 0.85 is default threshold
            total_semantic_penalty += semantic_penalty
            if semantic_penalty > 0:
                prompt_has_drift = True
                change_percent = (semantic_similarity - 0.85) / 0.85 * 100 # Rough percentage change from threshold
                prompt_drift_details["semantic_drift"] = {
                    "penalty": semantic_penalty,
                    "current_similarity": semantic_similarity,
                    "change_percent": change_percent
                }
                typer.echo(f"  Semantic Drift Penalty: {semantic_penalty:.2f} (Similarity: {semantic_similarity:.2f})")
        else:
            typer.echo("  Warning: Baseline semantic embedding not found. Skipping semantic drift check.")

        # --- Calculate Tone Drift ---
        tone_penalty = calculate_tone_drift_penalty(baseline_obj.extracted_metadata.tone, current_tone)
        total_tone_penalty += tone_penalty
        if tone_penalty > 0:
            prompt_has_drift = True
            prompt_drift_details["tone_drift"] = {
                "penalty": tone_penalty,
                "baseline_tone": baseline_obj.extracted_metadata.tone,
                "current_tone": current_tone
            }
            typer.echo(f"  Tone Drift Penalty: {tone_penalty:.2f} (Baseline: {baseline_obj.extracted_metadata.tone}, Current: {current_tone})")
        
        # --- Calculate Length Drift ---
        length_penalty = calculate_length_drift_penalty(baseline_obj.extracted_metadata.length, current_length)
        total_length_penalty += length_penalty
        if length_penalty > 0:
            prompt_has_drift = True
            change_percent = (current_length - baseline_obj.extracted_metadata.length) / baseline_obj.extracted_metadata.length * 100 if baseline_obj.extracted_metadata.length else (100 if current_length > 0 else 0)
            prompt_drift_details["length_drift"] = {
                "penalty": length_penalty,
                "baseline_length": baseline_obj.extracted_metadata.length,
                "current_length": current_length,
                "change_percent": change_percent
            }
            typer.echo(f"  Length Drift Penalty: {length_penalty:.2f} (Baseline: {baseline_obj.extracted_metadata.length}, Current: {current_length})")
        
        # --- Calculate Keyword Drift ---
        keyword_penalty = calculate_keyword_drift_penalty(baseline_obj.extracted_metadata.keywords, current_keywords)
        total_keyword_penalty += keyword_penalty
        if keyword_penalty > 0:
            prompt_has_drift = True
            lost_keywords = list(set(baseline_obj.extracted_metadata.keywords) - set(current_keywords))
            prompt_drift_details["keyword_loss"] = lost_keywords
            prompt_drift_details["penalty"] = keyword_penalty # Store penalty here for keyword drift
            typer.echo(f"  Keyword Drift Penalty: {keyword_penalty:.2f} (Lost: {', '.join(lost_keywords)})")

        if prompt_has_drift:
            overall_affected_prompts.append(baseline_obj.prompt_id)
            # Store prompt-specific drift details if needed for a more granular report later
            # For now, we're just accumulating penalties for overall score and listing affected prompts.

    if num_prompts_checked == 0:
        typer.echo("No prompts were checked. Exiting.")
        raise typer.Exit(code=0)

    # Average penalties across all checked prompts
    avg_semantic_penalty = total_semantic_penalty / num_prompts_checked
    avg_tone_penalty = total_tone_penalty / num_prompts_checked
    avg_length_penalty = total_length_penalty / num_prompts_checked
    avg_keyword_penalty = total_keyword_penalty / num_prompts_checked
    
    # Calculate overall health score
    health_score, health_status = calculate_health_score(
        avg_semantic_penalty,
        avg_tone_penalty,
        avg_length_penalty,
        avg_keyword_penalty
    )

    # Prepare drift details for the report
    report_drift_details: Dict[str, Any] = {}
    if avg_semantic_penalty > 0:
        # A simplified representation for overall report, actual values would need more storage
        report_drift_details["semantic_drift"] = {"change_percent": int(-avg_semantic_penalty * 100), "current_similarity": (1 - avg_semantic_penalty) * 0.85}
    if avg_tone_penalty > 0:
        report_drift_details["tone_drift"] = {"baseline_tone": "Mixed", "current_tone": "Changed"} # Cannot aggregate easily
    if avg_keyword_penalty > 0:
        report_drift_details["keyword_loss"] = ["Some keywords lost"] # Cannot aggregate easily
    if avg_length_penalty > 0:
        report_drift_details["length_drift"] = {"change_percent": int(avg_length_penalty * 100)}


    # Generate and save the run report
    report_content = generate_report_content(
        model_name=model_name,
        health_score=health_score,
        health_status=health_status,
        drift_details=report_drift_details,
        affected_prompts=list(set(overall_affected_prompts)) # Remove duplicates
    )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file_path = runs_dir / f"run_{run_timestamp}.json"
    with open(run_file_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": str(datetime.now()),
            "model_name": model_name,
            "health_score": health_score,
            "health_status": health_status,
            "drift_details": report_drift_details, # This is the aggregated one for the overall report
            "affected_prompts": list(set(overall_affected_prompts)),
            # More detailed prompt-by-prompt results could be stored here if needed
        }, f, indent=2)
    typer.echo(f"\nDrift check complete. Results saved to {run_file_path}")
    typer.echo("\n" + report_content) # Display report immediately

@app.command()
def report():
    """
    Displays the latest ModelPulse health report.
    """
    typer.echo("Fetching latest ModelPulse report...")
    runs_dir = BASE_DIR / "runs"
    if not runs_dir.exists():
        typer.echo("No run reports found. Please run 'modelpulse check' first.")
        raise typer.Exit(code=1)

    run_files = sorted(runs_dir.glob("run_*.json"), key=os.path.getmtime, reverse=True)
    if not run_files:
        typer.echo("No run reports found. Please run 'modelpulse check' first.")
        raise typer.Exit(code=1)

    latest_run_file = run_files[0]
    typer.echo(f"Loading report from: {latest_run_file}")

    with open(latest_run_file, "r", encoding="utf-8") as f:
        run_data = json.load(f)
    
    report_content = generate_report_content(
        model_name=run_data["model_name"],
        health_score=run_data["health_score"],
        health_status=run_data["health_status"],
        drift_details=run_data["drift_details"],
        affected_prompts=run_data["affected_prompts"]
    )
    typer.echo("\n" + report_content)


if __name__ == "__main__":
    app()
