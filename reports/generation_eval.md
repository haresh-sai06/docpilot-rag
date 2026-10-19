# Generation eval (qwen2.5-1.5b-local via ollama, n=25 questions, seed 13)

| Config | Faithfulness (NLI) | Answer relevance (cos) | Context hit rate@4 (gold passage in top-4 context) | p50 cache off | p95 cache off | p50 cache on (repeat) | p95 cache on (repeat) | Avg prompt tok | Avg completion tok |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid, no rerank | 0.427 | 0.904 | 0.76 | 5578 ms | 9656 ms | 0.10 ms | 0.10 ms | 750.9 | 64.5 |
| Hybrid + rerank | 0.416 | 0.918 | 0.88 | 9504 ms | 87395 ms | 0.10 ms | 0.10 ms | 758.1 | 77.9 |

## Per-stage latency, cache off (median over questions)

| Config | Retrieval (BM25+dense+fuse) | Rerank | LLM generate |
|---|---|---|---|
| Hybrid, no rerank | 66 ms | 0 ms | 5535 ms |
| Hybrid + rerank | 112 ms | 1380 ms | 6560 ms |

## Cost estimate per 1,000 queries (assumed price $0.15/1M input, $0.6/1M output tokens)

| Config | Cache off (measured tokens) | Cache on, repeat of the same questions (measured LLM calls) |
|---|---|---|
| Hybrid, no rerank | $0.1513 | $0.0000 |
| Hybrid + rerank | $0.1605 | $0.0000 |

All cold end-to-end latencies: p50 7297 ms, p95 50586 ms over 50 runs.
