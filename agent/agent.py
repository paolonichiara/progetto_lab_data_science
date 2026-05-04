import os
import json

from core.io import load_file
from core.eda.analyzer import analyze_file
from core.eda.stats import basic_stats
from core.eda.visualizer import generate_plots

from llm.explainer import Explainer


class EDAAgent:

    def __init__(self, data_path: str, output_dir: str = "outputs"):
        self.data_path = data_path
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.df = None
        self.analysis = None
        self.stats = None
        self.plots = None
        self.plots = None
        self.insight = None
        self.critic_review = None

        self.llm = Explainer()

    # =========================
    # LOAD
    # =========================
    def load(self):
        self.df = load_file(self.data_path)
        return self.df

    # =========================
    # ANALYZER
    # =========================
    def run_analyzer(self):
        result = analyze_file(self.data_path)

        self.analysis = result["report"]
        return self.analysis

    # =========================
    # STATS
    # =========================
    def run_stats(self):
        self.stats = basic_stats(self.df, self.analysis["roles"])
        return self.stats

    # =========================
    # VISUALIZER
    # =========================
    def run_visualizer(self):
        self.plots = generate_plots(
            self.df,
            self.analysis["roles"],
            output_dir=os.path.join(self.output_dir, "plots")
        )
        return self.plots

    # =========================
    # LLM (EXPLAINER + CRITIC)
    # =========================
    def run_llm(self):

        print("  -> Explainer Agent is thinking...")
        self.insight = self.llm.explain(
            report=self.analysis,
            stats=self.stats
        )

        print("  -> Critic Agent is reviewing...")
        self.critic_review = self.llm.criticize(
            insight=self.insight,
            report=self.analysis,
            stats=self.stats
        )

        return self.insight, self.critic_review

    # =========================
    # PIPELINE
    # =========================
    def run(self):

        print("\n[1/5] Load...")
        self.load()

        print("\n[2/5] Analyzer...")
        self.run_analyzer()

        print("\n[3/5] Stats...")
        self.run_stats()

        print("\n[4/5] Visualizer...")
        self.run_visualizer()

        print("\n[5/5] LLM Agents (Explainer + Critic)...")
        self.run_llm()

        return self.insight, self.critic_review

    # =========================
    # EXPORT
    # =========================
    def export_report(self):

        path = os.path.join(self.output_dir, "report.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "analysis": self.analysis,
                "stats": self.stats,
                "plots": self.plots,
                "insight": self.insight,
                "critic_review": self.critic_review
            }, f, indent=2, ensure_ascii=False)

        return path