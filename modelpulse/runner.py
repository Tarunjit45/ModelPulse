import requests
from typing import Dict, Any, Optional

class LLMRunner:
    """
    Handles interaction with the local Ollama LLM.
    """
    def __init__(self, model_name: str = "llama3", ollama_base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.session = requests.Session()

    def _call_ollama(self, prompt: str, model_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Internal method to call the Ollama API.
        """
        url = f"{self.ollama_base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": model_config or {}
        }
        
        try:
            response = self.session.post(url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to Ollama server at {self.ollama_base_url}.")
            print("Please ensure Ollama is running and the model is available.")
            return f"ERROR: Ollama connection failed. Is {self.model_name} pulled and running?"
        except requests.exceptions.Timeout:
            print("Error: Ollama request timed out.")
            return "ERROR: Ollama request timed out."
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return f"ERROR: Ollama API call failed: {e}"

    def generate_response(self, prompt: str, model_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a response from the LLM for a given prompt.
        """
        return self._call_ollama(prompt, model_config)

# --- Mocking for testing/offline use ---
class MockLLMRunner(LLMRunner):
    def __init__(self, model_name: str = "mock-model", ollama_base_url: str = "mock-url"):
        super().__init__(model_name, ollama_base_url)
        self.mock_responses = {
            "Explain how to reset my password.": "To reset your password, navigate to the account settings page, select 'Forgot Password', and follow the instructions to set a new one. This is a step-by-step guide.",
            "Describe the capital of France.": "The capital of France is Paris, known for its art, fashion, gastronomy, and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine. Beyond such landmarks as the Eiffel Tower and the 12th-century, Gothic Notre-Dame Cathedral, the city is known for its cafe culture and designer boutiques along the Rue du Faubourg Saint-Honoré."
        }
        print("WARNING: Using MockLLMRunner. No actual LLM calls will be made.")

    def _call_ollama(self, prompt: str, model_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Returns a mock response for the given prompt.
        """
        response = self.mock_responses.get(prompt.strip(), f"Mock response for: {prompt}")
        print(f"MockLLMRunner returning: {response[:50]}...")
        return response

if __name__ == "__main__":
    # Example usage:
    # try:
    #     runner = LLMRunner(model_name="llama3")
    #     test_prompt = "What is the capital of Canada?"
    #     response = runner.generate_response(test_prompt)
    #     print(f"Prompt: {test_prompt}\nResponse: {response}")
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    # Example of mock usage:
    mock_runner = MockLLMRunner()
    mock_prompt_1 = "Explain how to reset my password."
    mock_response_1 = mock_runner.generate_response(mock_prompt_1)
    print(f"Mock Prompt 1: {mock_prompt_1}\nMock Response 1: {mock_response_1}")

    mock_prompt_2 = "What is the capital of France?"
    mock_response_2 = mock_runner.generate_response(mock_prompt_2)
    print(f"Mock Prompt 2: {mock_prompt_2}\nMock Response 2: {mock_response_2}")

    mock_prompt_3 = "A new unknown prompt."
    mock_response_3 = mock_runner.generate_response(mock_prompt_3)
    print(f"Mock Prompt 3: {mock_prompt_3}\nMock Response 3: {mock_response_3}")
