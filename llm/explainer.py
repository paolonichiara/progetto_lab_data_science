from llm.client import OllamaClient
from llm.prompts import build_prompt, build_critic_prompt


class Explainer:
    def __init__(self, model: str = "llama3.2:latest"):
        self.client = OllamaClient(model=model)

    def explain(self, report: dict, stats: dict = None, question: str = None, context: str = "") -> str:

        # report può essere wrapper o già flat
        report_data = report.get("report", report)

        dataset_type = report_data.get("dataset_type", "unknown")

        prompt = build_prompt(
            report=report_data,
            stats=stats or {},
            context=context or "",
            question=question,
            dataset_type=dataset_type
        )

        return self.client.generate(prompt)

    def criticize(self, insight: str, report: dict, stats: dict = None) -> str:
        report_data = report.get("report", report)
        prompt = build_critic_prompt(
            insight=insight,
            report=report_data,
            stats=stats or {}
        )
        return self.client.generate(prompt)