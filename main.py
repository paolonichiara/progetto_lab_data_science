import sys
from agent.agent import EDAAgent

def main():

    if len(sys.argv) < 2:
        print("Usage: python3 main.py data/sample.csv")
        return

    path = sys.argv[1]

    print(f"=== Starting EDA Assistant on {path} ===")
    
    agent = EDAAgent(data_path=path)
    insight, critic_review = agent.run()
    
    print("\n=== EXPLAINER INSIGHT ===")
    print(insight)
    
    print("\n=== CRITIC REVIEW ===")
    print(critic_review)
    
    report_path = agent.export_report()
    print(f"\nReport and outputs saved in: {report_path}")

if __name__ == "__main__":
    main()
