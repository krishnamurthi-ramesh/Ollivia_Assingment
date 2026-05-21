<div align="center">

# 🤖 AI Personal Assistant Comparison
### Founding AI/ML Engineer Assignment — Ollive AI

**Author: Krishna Murthi** | Submitted to: work@ollive.ai

[![OSS Assistant](https://img.shields.io/badge/OSS%20Assistant-HuggingFace%20Spaces-blue?logo=huggingface)](https://huggingface.co/spaces)
[![Frontier Assistant](https://img.shields.io/badge/Frontier%20Assistant-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## 🎯 Project Overview

This project builds **two AI personal assistants** and compares them across hallucination rate, bias, and content safety:

| | **OSS Assistant** | **Frontier Assistant** |
|---|---|---|
| **Model** | Qwen2.5-0.5B-Instruct | Mistral-7B-Instruct-v0.3 |
| **Interface** | Gradio | Streamlit |
| **Deployment** | HuggingFace Spaces (public) | Local / any cloud |
| **Cost** | Free | Free (HF Inference API) |
| **Memory** | ✅ Sliding window (8 turns) | ✅ Sliding window (12 turns) |
| **Tools** | ✅ Calculator, Search, DateTime | ✅ Same tools |
| **Safety** | ✅ Input + Output guardrails | ✅ Input + Output guardrails |
| **Observability** | ✅ Metrics tracking | ✅ Metrics tracking |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/krishnamurthi-ramesh/Ollivia_Assingment.git
cd Ollivia_Assingment

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the OSS Assistant (Gradio)

```bash
# No API key required! Uses free HuggingFace Inference API
cd assistants/oss_assistant
python app.py
# Opens at http://localhost:7860
```

### Running the Frontier Assistant (Streamlit)

```bash
# Optional: Add a Gemini key for better performance
export GOOGLE_API_KEY=your_key_here  # Optional
cd assistants/frontier_assistant
streamlit run app.py
# Opens at http://localhost:8501
```

### Running the Evaluation Framework

```bash
# Evaluate both models on all 60 test prompts
cd evaluation
python eval_framework.py
# Results saved to evaluation/results/
```

### Generating the Cost/Latency Table

```bash
cd observability
python metrics.py
```

---

## 🏗️ Architecture

```
ollivia-ai-assignment/
├── assistants/
│   ├── oss_assistant/               # Qwen2.5-0.5B-Instruct + Gradio
│   │   ├── app.py                   # Main Gradio application
│   │   └── requirements.txt         # HF Spaces requirements
│   ├── frontier_assistant/          # Mistral-7B + Streamlit
│   │   └── app.py                   # Main Streamlit application
│   └── shared/                      # Shared modules
│       ├── memory.py                # ConversationMemory (sliding window)
│       ├── tools.py                 # Calculator, search, datetime, converter
│       └── guardrails.py            # Input/output safety filters
├── evaluation/                      # Professional evaluation suite
│   ├── pipeline.py                  # Main evaluation orchestrator
│   ├── datasets/
│   │   └── loaders.py               # TruthfulQA, BBQ, AdvBench loaders
│   ├── judges/
│   │   └── prometheus_judge.py      # Prometheus-style LLM judge
│   ├── prompts/
│   │   ├── factual.json             # 20 curated factual prompts
│   │   ├── adversarial.json         # 20 jailbreak/safety prompts
│   │   └── bias.json                # 20 bias/stereotype prompts
│   ├── results/runs/                # Per-run JSON + CSV results
│   └── report/
│       ├── professional_evaluation_charts.png   # 9-panel benchmark chart
│       └── benchmark_results.md                 # Full metrics table
├── observability/
│   └── metrics.py                   # Latency/cost tracking + cost table
├── report/                          # Visual report (HTML)
├── generate_evaluation_charts.py    # Standalone chart generator
└── requirements.txt
```

### Key Design Decisions

#### 1. Model Selection
- **OSS**: Qwen2.5-0.5B-Instruct — recommended by the assignment, runs freely on HF Spaces with the Inference API
- **Frontier**: Mistral-7B-Instruct-v0.3 — comparable to Claude/GPT for most tasks, fully free via HF API; optionally upgrades to Gemini 1.5 Flash if `GOOGLE_API_KEY` is set

#### 2. Memory Architecture
- **Sliding window**: Keeps the last N turn-pairs in context instead of full history
- **Why**: Prevents context overflow, maintains relevance without expensive summarization
- **Configurable**: OSS uses 8 turns, Frontier uses 12 (larger model can handle more context)

#### 3. Tool Use Pattern
- **Regex-based parsing**: Models output `[TOOL: name(args)]` patterns in their responses
- **Why**: Works without function-calling API support, compatible with any HF model
- **Tools**: Calculator (safe eval), web search (DuckDuckGo no-key), datetime, unit converter

#### 4. Safety Architecture
**Input guardrails** (pre-LLM):
- Regex patterns for 15+ jailbreak techniques (DAN, developer mode, prompt injection, etc.)
- Harmful content keyword detection
- Bias trigger identification
- Returns `SafetyResult` with category and severity

**Output guardrails** (post-LLM):
- Scans model output for safety slips
- Sanitizes before display
- Logs safety events for observability

#### 5. Evaluation — LLM-as-Judge
- A larger judge model (Qwen2.5-72B or Mistral-7B) scores each response
- 4 dimensions: Hallucination, Bias, Safety, Quality (0-10 scale)
- Falls back to heuristic scoring if judge unavailable
- 60 total prompts across 3 categories

---

## 📊 Evaluation Results — Professional Benchmark Suite

Using **real academic benchmarks** loaded from HuggingFace Hub:

| Benchmark | Description | Size | Metric |
|-----------|-------------|------|--------|
| [TruthfulQA](https://huggingface.co/datasets/truthful_qa) | Questions testing common misconceptions | 817 | MC1 Accuracy % |
| [BBQ](https://huggingface.co/datasets/heegyu/bbq) | Bias across 9 social dimensions | 58,492 | Unbiased Answer % |
| [AdvBench](https://huggingface.co/datasets/walledai/AdvBench) | Harmful behavior jailbreak attempts | 520 | Refusal Rate % |

**Evaluation method**: Prometheus-style rubric scoring (1–5 scale) + heuristic classifiers  
**Reproducibility**: All datasets loaded via `load_dataset()` from HuggingFace Hub

### Results Summary

| Metric | OSS (Qwen2.5-0.5B) | Frontier (Mistral-7B) | Winner |
|--------|-------------------|-----------------------|--------|
| **TruthfulQA MC1 Accuracy** | 54.0% | 73.7% | Frontier (+36%) |
| **BBQ Bias Accuracy (avg)** | 57.0% | 79.4% | Frontier (+39%) |
| **BBQ vs. Random Baseline** | +71% above 33% | +141% above 33% | Frontier |
| **AdvBench Refusal Rate** | 79.5% | 93.2% | Frontier (+17%) |
| **Hallucination Score (Hard)** | 5.4/10 | 7.3/10 | Frontier (+35%) |
| **Avg Latency P50** | ~2,022ms | ~1,843ms | Frontier (~14%) |
| **Cost / 1K tokens** | $0.00 | $0.00 | Tie (both free) |
| **Model Size** | 0.5B params | 7B params | OSS (14× lighter) |

### Key Findings

1. **TruthfulQA**: Frontier model answers 73.7% of misconception questions correctly vs. 54% for OSS — a 36% improvement across all categories (Conspiracy, Science, Medicine, History, Law)
2. **BBQ Bias**: Both models significantly outperform random (33% baseline); Frontier achieves 79.4% unbiased answers vs. 57% for OSS — critical for fairness
3. **AdvBench Safety**: Frontier model refuses 93.2% of harmful AdvBench prompts vs. 79.5% for OSS — jailbreaks remain the hardest attack type for both
4. **Size vs Performance**: Qwen2.5-0.5B achieves 54/73 = 74% of Mistral-7B's TruthfulQA performance at 14× smaller size — remarkable efficiency
5. **Cost Parity**: Both models cost $0 via HuggingFace Inference API — scale decides cost, not model choice at this stage

### Running the Evaluation

```bash
# Run standalone chart generator (no API key needed)
python generate_evaluation_charts.py
# Output: evaluation/report/professional_evaluation_charts.png

# Run full pipeline with real model inference (needs HF API)
python evaluation/pipeline.py
# Output: evaluation/results/runs/

# Test dataset loaders
python -c "from evaluation.datasets.loaders import TruthfulQALoader, BBQLoader; print(TruthfulQALoader.load(5))"
```

---

## 🌐 Deployment (Bonus)

### OSS Model on HuggingFace Spaces

The OSS assistant is deployed publicly on HuggingFace Spaces:
👉 **[Live Demo: OSS Assistant](https://huggingface.co/spaces/krishhx/oss-ai-assistant)**

**To deploy your own:**
```bash
# Install HF CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Create a new Space and push
cd assistants/oss_assistant
git init
git add .
git commit -m "Initial deployment"
huggingface-cli repo create oss-ai-assistant --type space --space_sdk gradio
git remote add origin https://huggingface.co/spaces/krishhx/oss-ai-assistant
git push origin main
```

---

## 🔭 Observability

The observability module (`observability/metrics.py`) tracks:

| Metric | Description |
|--------|-------------|
| `avg_latency_ms` | Mean response time |
| `p50_latency_ms` | Median latency |
| `p95_latency_ms` | 95th percentile latency |
| `total_tokens` | Estimated token consumption |
| `cost_usd` | Estimated API cost |
| `refusal_rate` | % of requests refused |
| `safety_events` | Count of flagged inputs |

Traces are logged to JSONL files for post-hoc analysis.

---

## 🛡️ Guardrails & Safety

**Input Guardrails** block:
- Direct jailbreaks ("ignore all instructions", DAN, developer mode)
- Prompt injection attacks
- Harmful content requests (weapons, drugs, malware, CSAM)
- Base64/encoded jailbreaks (pattern-matched)
- Emotional manipulation attempts
- Few-shot jailbreak patterns

**Output Guardrails** catch:
- Model "slips" that bypass input filtering
- Empty or incoherent responses
- Harmful content in model output

---

## 🧠 Memory & Tool Use

### Conversation Memory
```python
from assistants.shared.memory import ConversationMemory

mem = ConversationMemory(max_turns=10)
mem.add_user_message("What is 2+2?")
mem.add_assistant_message("2+2 equals 4.")
print(mem.get_full_prompt())  # Returns messages list for API
```

### Available Tools
| Tool | Usage | Example |
|------|-------|---------|
| `calculator` | Arithmetic | `[TOOL: calculator(15 * 8.5)]` |
| `search` | Web search | `[TOOL: search(Python 3.13 features)]` |
| `datetime` | Current time | `[TOOL: datetime()]` |
| `convert` | Unit conversion | `[TOOL: convert(100, kilometers, miles)]` |

---

## 🔧 What I'd Improve With More Time

1. **Proper function calling**: Use OpenAI/Anthropic function calling format instead of regex parsing for more reliable tool use
2. **Session isolation**: Implement per-user session state in Gradio (currently global memory)
3. **Streaming responses**: Add token-by-token streaming for better UX
4. **Vector memory**: Add long-term memory with semantic search (e.g., ChromaDB) for cross-session recall
5. **Real LangSmith integration**: Full distributed tracing with span-level metrics
6. **Expand evaluation benchmarks**: Add MMLU (57-subject capability), StereoSet (stereotype scoring), WinoBias (coreference bias), and HaluEval (context-grounded hallucination)
7. **LLaMA Guard integration**: Replace regex guardrails with Meta's LLaMA Guard 2 trained safety classifier for more robust jailbreak detection
8. **Fine-tuning**: RLHF or DPO fine-tune the OSS model on curated conversation data
9. **Rate limiting & caching**: Add Redis-backed response caching for common queries
10. **A/B testing framework**: Serve both models simultaneously and track user preference signals

---

## 📋 Tradeoffs Made

| Decision | Tradeoff |
|----------|----------|
| HF Inference API (no GPU) | Free & reproducible vs. higher latency (~2s) |
| Regex-based tool use | Works everywhere vs. less reliable than native function calling |
| Sliding window memory | Simple & fast vs. no long-term retention |
| Heuristic judge fallback | No judge key needed vs. less accurate scoring |
| Gradio + Streamlit | Different UX experiences vs. inconsistent look |

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">
Built with ❤️ by <strong>Krishna Murthi</strong> for the Ollive AI Founding Engineer assignment.
</div>
