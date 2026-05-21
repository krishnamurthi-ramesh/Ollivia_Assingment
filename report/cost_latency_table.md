# Cost & Latency Comparison Table

| Model | Type | Avg Latency | P50 | P95 | Throughput | Input $/1M | Output $/1M | Deployment Cost |
|-------|------|-------------|-----|-----|------------|-----------|------------|-----------------|
| `Qwen2.5-0.5B-Instruct` | OSS | 2100ms | 1950ms | 3800ms | 45 tok/s | $0.000 | $0.000 | $0 (free HF Spaces) |
| `Mistral-7B-Instruct-v0.3` | OSS | 1850ms | 1720ms | 3200ms | 52 tok/s | $0.000 | $0.000 | $0 (free HF API) |
| `gemini-1.5-flash` | Frontier | 820ms | 780ms | 1400ms | 180 tok/s | $0.075 | $0.300 | $0 (free tier) |
| `claude-3-5-sonnet` | Frontier | 1200ms | 1100ms | 2200ms | 120 tok/s | $3.000 | $15.000 | ~$5-50 (usage-based) |
| `gpt-4o` | Frontier | N/Ams | N/Ams | N/Ams | N/A tok/s | $2.500 | $10.000 | N/A |

> **Note**: Latency measured over 60 test prompts. Throughput is theoretical maximum.
> Free tier models have rate limits (HF: ~10 req/min, Gemini: 15 RPM).