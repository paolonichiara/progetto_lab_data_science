import requests


class OllamaClient:
    """
    Client per LLM locale (Ollama).
    """

    def __init__(self, model: str = "llama3.2:latest"):
        # Llama 3.2 (3B) è il bilanciamento perfetto tra intelligenza e leggerezza per Mac Intel
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )

            data = response.json()

            if "response" in data:
                return data["response"]

            if "message" in data and "content" in data["message"]:
                return data["message"]["content"]

            if "error" in data:
                return f"Ollama error: {data['error']}"

            return str(data)

        except Exception as e:
            return f"LLM error: {e}"