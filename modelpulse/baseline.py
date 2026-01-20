from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, Literal, Optional

class ModelConfig(BaseModel):
    """
    Represents the configuration of the LLM used for generating outputs.
    """
    temperature: float = Field(default=0.7, description="Model temperature setting.")
    top_p: float = Field(default=0.9, description="Model top-p setting for nucleus sampling.")
    # Add other common model parameters as needed

class ExtractedMetadata(BaseModel):
    """
    Stores extracted metadata from an LLM's output.
    """
    tone: Literal["instructive", "neutral", "speculative"] # Enforce specific tone labels
    length: int # Number of words or tokens
    keywords: list[str] # List of key nouns extracted
    semantic_embedding: Optional[list[float]] = None # Sentence embedding as a list of floats

class Baseline(BaseModel):
    """
    Represents a baseline output for a critical prompt, serving as a reference
    for healthy model behavior.
    """
    prompt_id: str = Field(..., description="Unique identifier for the prompt.")
    prompt_text: str = Field(..., description="The original prompt text.")
    model_name: str = Field(..., description="Name of the LLM used (e.g., 'llama3').")
    model_config: ModelConfig = Field(default_factory=ModelConfig, description="Configuration of the model at baseline creation.")
    baseline_output: str = Field(..., description="The LLM's healthy output for the prompt.")
    extracted_metadata: ExtractedMetadata = Field(..., description="Metadata extracted from the baseline output.")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of when the baseline was created.")

    def to_json_file(self, file_path: Path):
        """
        Saves the baseline to a JSON file.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, file_path: Path):
        """
        Loads a baseline from a JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())
