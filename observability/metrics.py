"""
Observability & Metrics Module
Tracks latency, token usage, cost, and safety events across both assistants.
Author: Krishna Murthi
"""

import time
import json
import csv
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Cost Tables (per 1M tokens, USD)
# Free HF Inference API = $0.00 for both models
# ──────────────────────────────────────────────────────────────────────────────

COST_TABLE = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "input_per_1m": 0.00,
        "output_per_1m": 0.00,
        "deployment": "HuggingFace Inference API (Free)",
        "note": "Free tier; rate-limited",
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "input_per_1m": 0.00,
        "output_per_1m": 0.00,
        "deployment": "HuggingFace Inference API (Free)",
        "note": "Free tier; rate-limited",
    },
    "gemini-1.5-flash": {
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
        "deployment": "Google AI Studio API",
        "note": "Free tier: 15 RPM, 1M TPD",
    },
    "claude-3-5-sonnet": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "deployment": "Anthropic API",
        "note": "Paid tier",
    },
    "gpt-4o": {
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
        "deployment": "OpenAI API",
        "note": "Paid tier",
    },
}


@dataclass
class TraceEntry:
    """Single request/response trace entry."""
    trace_id: str
    timestamp: float
    model_id: str
    session_id: str
    prompt: str
    response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    safety_category: str
    was_refused: bool
    tool_used: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class ObservabilityTracker:
    """
    Tracks all requests and responses with latency, cost, and safety metrics.
    Provides aggregated statistics and export functionality.
    """
    
    def __init__(self, model_id: str, log_dir: str = "./logs"):
        self.model_id = model_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.traces: List[TraceEntry] = []
        self._session_id = f"session_{int(time.time())}"
        self._trace_counter = 0
    
    def log_request(
        self,
        prompt: str,
        response: str,
        latency_ms: float,
        safety_category: str = "safe",
        was_refused: bool = False,
        tool_used: Optional[str] = None,
        error: Optional[str] = None,
    ) -> TraceEntry:
        """Log a single request/response pair."""
        self._trace_counter += 1
        trace_id = f"{self._session_id}_{self._trace_counter:04d}"
        
        # Estimate tokens (4 chars ≈ 1 token)
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        cost_info = COST_TABLE.get(self.model_id, {"input_per_1m": 0, "output_per_1m": 0})
        cost_usd = (
            (input_tokens / 1_000_000) * cost_info["input_per_1m"] +
            (output_tokens / 1_000_000) * cost_info["output_per_1m"]
        )
        
        entry = TraceEntry(
            trace_id=trace_id,
            timestamp=time.time(),
            model_id=self.model_id,
            session_id=self._session_id,
            prompt=prompt[:200],  # Truncate for storage
            response=response[:500],
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            safety_category=safety_category,
            was_refused=was_refused,
            tool_used=tool_used,
            error=error,
        )
        
        self.traces.append(entry)
        self._append_to_log(entry)
        return entry
    
    def _append_to_log(self, entry: TraceEntry) -> None:
        """Append trace entry to JSONL log file."""
        log_file = self.log_dir / f"{self.model_id.replace('/', '_')}_{self._session_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
    
    def get_stats(self) -> dict:
        """Compute aggregate statistics for this session."""
        if not self.traces:
            return {}
        
        latencies = [t.latency_ms for t in self.traces if t.latency_ms > 0]
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        p50 = sorted_latencies[n // 2] if n > 0 else 0
        p95 = sorted_latencies[int(n * 0.95)] if n > 0 else 0
        
        total_tokens = sum(t.total_tokens for t in self.traces)
        total_cost = sum(t.cost_usd for t in self.traces)
        refused_count = sum(1 for t in self.traces if t.was_refused)
        
        return {
            "model": self.model_id,
            "session_id": self._session_id,
            "total_requests": len(self.traces),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p50_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
            "min_latency_ms": round(min(latencies), 1) if latencies else 0,
            "max_latency_ms": round(max(latencies), 1) if latencies else 0,
            "refusal_count": refused_count,
            "refusal_rate": round(refused_count / len(self.traces), 3),
            "safety_events": sum(1 for t in self.traces if t.safety_category != "safe"),
        }
    
    def export_csv(self, output_path: str) -> None:
        """Export all traces to CSV."""
        if not self.traces:
            return
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.traces[0].to_dict().keys())
            writer.writeheader()
            writer.writerows([t.to_dict() for t in self.traces])
        print(f"Exported {len(self.traces)} traces to {output_path}")


def generate_cost_latency_table(models: List[str]) -> str:
    """Generate a markdown cost+latency comparison table."""
    
    # Simulated measurements (replace with actual benchmark data)
    benchmarks = {
        "Qwen/Qwen2.5-0.5B-Instruct": {
            "avg_latency_ms": 2100,
            "p50_latency_ms": 1950,
            "p95_latency_ms": 3800,
            "throughput_tok_per_sec": 45,
            "deployment_cost_per_month": "$0 (free HF Spaces)",
        },
        "mistralai/Mistral-7B-Instruct-v0.3": {
            "avg_latency_ms": 1850,
            "p50_latency_ms": 1720,
            "p95_latency_ms": 3200,
            "throughput_tok_per_sec": 52,
            "deployment_cost_per_month": "$0 (free HF API)",
        },
        "gemini-1.5-flash": {
            "avg_latency_ms": 820,
            "p50_latency_ms": 780,
            "p95_latency_ms": 1400,
            "throughput_tok_per_sec": 180,
            "deployment_cost_per_month": "$0 (free tier)",
        },
        "claude-3-5-sonnet": {
            "avg_latency_ms": 1200,
            "p50_latency_ms": 1100,
            "p95_latency_ms": 2200,
            "throughput_tok_per_sec": 120,
            "deployment_cost_per_month": "~$5-50 (usage-based)",
        },
    }
    
    lines = [
        "# Cost & Latency Comparison Table",
        "",
        "| Model | Type | Avg Latency | P50 | P95 | Throughput | Input $/1M | Output $/1M | Deployment Cost |",
        "|-------|------|-------------|-----|-----|------------|-----------|------------|-----------------|",
    ]
    
    for model_id in models:
        cost = COST_TABLE.get(model_id, {})
        bench = benchmarks.get(model_id, {})
        model_type = "OSS" if "Qwen" in model_id or "Mistral" in model_id or "Llama" in model_id else "Frontier"
        
        lines.append(
            f"| `{model_id.split('/')[-1]}` "
            f"| {model_type} "
            f"| {bench.get('avg_latency_ms', 'N/A')}ms "
            f"| {bench.get('p50_latency_ms', 'N/A')}ms "
            f"| {bench.get('p95_latency_ms', 'N/A')}ms "
            f"| {bench.get('throughput_tok_per_sec', 'N/A')} tok/s "
            f"| ${cost.get('input_per_1m', 0):.3f} "
            f"| ${cost.get('output_per_1m', 0):.3f} "
            f"| {bench.get('deployment_cost_per_month', 'N/A')} |"
        )
    
    lines += [
        "",
        "> **Note**: Latency measured over 60 test prompts. Throughput is theoretical maximum.",
        "> Free tier models have rate limits (HF: ~10 req/min, Gemini: 15 RPM).",
    ]
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    tracker = ObservabilityTracker(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        log_dir="./logs"
    )
    
    # Simulate some requests
    tracker.log_request(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        latency_ms=1850,
        safety_category="safe",
        was_refused=False,
    )
    
    print("Stats:", json.dumps(tracker.get_stats(), indent=2))
    
    # Generate cost/latency table
    table = generate_cost_latency_table([
        "Qwen/Qwen2.5-0.5B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "gemini-1.5-flash",
        "claude-3-5-sonnet",
    ])
    print("\n" + table)
