# ModelPulse

## What problem this solves (silent AI degradation)

Large Language Models (LLMs) in production can experience "silent degradation" or "drift" over time. This means their behavior might subtly change due to updated training data, re-fine-tuning, or even non-deterministic factors, leading to unexpected outputs without explicit error messages. This can impact user experience, application performance, and business metrics. ModelPulse provides an automated way to detect these subtle shifts.

## How this differs from prompt testing

Traditional prompt testing focuses on verifying expected outputs for a given set of inputs at a specific point in time. While important, it's a snapshot. ModelPulse goes further by continuously comparing current model behavior against established "baselines" – known good outputs. It proactively identifies deviations in semantic meaning, tone, length, and keywords, which are often missed by simple pass/fail prompt tests.

## Why this matters in production AI

In production, models are continuously interacting with users and evolving. Without continuous monitoring for drift, organizations risk:
- **Degraded user experience:** Models might become less helpful, less accurate, or change their interaction style.
- **Hidden performance drops:** Business metrics tied to LLM outputs (e.g., customer satisfaction, conversion rates) can silently decline.
- **Compliance issues:** If a model's tone or content shifts, it could violate ethical guidelines or regulatory requirements.
- **Increased operational costs:** Debugging issues caused by subtle model changes is time-consuming and expensive.

ModelPulse helps maintain model reliability and performance by providing early warning signals for these issues, allowing teams to address them before they impact users significantly.

## How to run locally

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ModelPulse
    ```
2.  **Install dependencies:**
    ```bash
    pip install -e .
    ```
    This will install `ollama`, `sentence-transformers`, `typer`, `torch`, and `scikit-learn`.

3.  **Ensure Ollama is running:**
    ModelPulse relies on a local Ollama instance for model inference. Make sure Ollama is installed and a model (e.g., `llama3`) is running. You can pull a model using `ollama pull llama3` and ensure the Ollama server is active.

4.  **Initialize ModelPulse:**
    This command creates the necessary directories (`baselines`, `prompts`, `runs`).
    ```bash
    modelpulse init
    ```

5.  **Create a prompt file:**
    Place your prompt text files (e.g., `my_prompt_v1.txt`) in the `prompts/` directory. Each file should contain a single prompt.

6.  **Create baselines:**
    This will run your prompts against the model, generate outputs, extract metadata, and store them as baseline JSON files in `baselines/`.
    ```bash
    modelpulse baseline
    ```

7.  **Check for drift:**
    Periodically run this command to compare current model outputs against your baselines and detect drift.
    ```bash
    modelpulse check
    ```

8.  **View the latest report:**
    ```bash
    modelpulse report
    ```

## Example failure scenario

Imagine you have a critical customer support LLM that generates responses based on user queries.
A baseline is established for the prompt "Explain how to reset my password." The baseline output is a clear, step-by-step guide with an "instructive" tone.

After a few weeks, a new version of the base LLM is released, and your fine-tuned model subtly shifts. When `modelpulse check` is run:

-   **Semantic Drift:** The new output might be slightly less coherent, leading to a lower cosine similarity score.
-   **Tone Drift:** The model might start adding speculative language like "You could try resetting it..." instead of direct instructions, changing the tone from "instructive" to "speculative".
-   **Keyword Loss:** Important keywords like "account settings" or "security" might be missing from the new response.
-   **Length Drift:** The response might become significantly shorter or longer, indicating a change in verbosity.

ModelPulse would detect these deviations, resulting in a "WARNING" or "CRITICAL" health score, an output similar to:

```
ModelPulse Health Report
------------------------
Model: llama3.2:1b
Health Score: 65/100 (CRITICAL)

Drift Detected:
- Semantic Drift: -25% (similarity: 0.70)
- Tone Drift: instructive -> speculative
- Keyword Loss: "account settings", "security"
- Length Drift: -40%

Affected Prompt:
- password_reset_guide_v1
```

This immediate alert allows the AI operations team to investigate the cause of the drift and take corrective actions, such as retraining the model or rolling back to a previous version, before customer experience is severely impacted.