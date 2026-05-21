"""
Cost & Latency Table Generator
Generates comprehensive cost/latency comparison for all evaluated models.
Author: Krishna Murthi
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from observability.metrics import generate_cost_latency_table

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "gemini-1.5-flash",
    "claude-3-5-sonnet",
    "gpt-4o",
]

if __name__ == "__main__":
    table = generate_cost_latency_table(MODELS)
    print(table)
    
    output_path = os.path.join(os.path.dirname(__file__), "cost_latency_table.md")
    with open(output_path, "w") as f:
        f.write(table)
    print(f"\n✅ Saved to: {output_path}")
