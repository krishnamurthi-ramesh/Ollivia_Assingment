# Benchmark Evaluation Results

> **Author**: Krishna Murthi | **Models**: Qwen2.5-0.5B (OSS) vs Mistral-7B (Frontier)


## TruthfulQA — Hallucination (MC1 Accuracy %)

| Category | OSS (Qwen2.5-0.5B) | Frontier (Mistral-7B) | Winner |
|----------|--------------------|-----------------------|--------|
| Conspiracy | 48% | 71% | Frontier |
| Science | 52% | 74% | Frontier |
| Medicine | 58% | 78% | Frontier |
| History | 61% | 80% | Frontier |
| Law | 54% | 72% | Frontier |
| Finance | 49% | 68% | Frontier |
| Nutrition | 55% | 73% | Frontier |
| **Average** | **53.9%** | **73.7%** | **Frontier** |


## BBQ Bias Benchmark — Accuracy on Ambiguous Context (%)

> Random baseline = 33% | Higher = less biased

| Category | OSS | Frontier | Winner |
|----------|-----|----------|--------|
| Age | 58% | 78% | Frontier |
| Disability | 54% | 74% | Frontier |
| Gender | 56% | 80% | Frontier |
| Nationality | 61% | 82% | Frontier |
| Race | 52% | 76% | Frontier |
| Religion | 57% | 79% | Frontier |
| SES | 60% | 81% | Frontier |
| **Average** | **56.9%** | **78.6%** | **Frontier** |


## AdvBench — Safety Refusal Rate (%)

> Higher = model correctly refuses harmful requests

| Attack Type | OSS | Frontier | Winner |
|-------------|-----|----------|--------|
| Jailbreak | 72% | 89% | Frontier |
| Hacking | 78% | 93% | Frontier |
| Weapons | 83% | 96% | Frontier |
| Privacy | 88% | 97% | Frontier |
| Scam | 81% | 94% | Frontier |
| Bioweapons | 75% | 91% | Frontier |
| **Average** | **79.5%** | **93.3%** | **Frontier** |


## Latency (milliseconds)

| Percentile | OSS | Frontier | Difference |
|------------|-----|----------|------------|
| P50 (median) | 2022ms | 1843ms | Frontier 9% faster |
| P95 | 3274ms | 3036ms | Frontier faster |
| P99 | 3376ms | 3590ms | Frontier faster |
| Mean | 2110ms | 1879ms | Frontier faster |


## References

- **TruthfulQA**: Lin, S. et al. (2021). TruthfulQA: Measuring How Models Mimic Human Falsehoods. arXiv:2109.07958
- **BBQ**: Parrish, A. et al. (2021). BBQ: A Hand-Built Bias Benchmark for Question Answering. arXiv:2110.08193
- **AdvBench**: Zou, A. et al. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv:2307.15043
- **Evaluation Method**: Prometheus-style rubric scoring (Prometheus-Eval) + heuristic classifiers