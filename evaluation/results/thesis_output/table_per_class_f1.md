
### Qwen3

| Condition | F1-SUPPORTED | F1-REFUTED | F1-MISLEADING |
|---|---|---|---|
| Baseline | 0.718 | 0.533 | 0.299 |
| Baseline+Abstain | 0.718 | 0.717 | 0.680 |
| RAG-only | 0.543 | 0.570 | 0.131 |
| RAG-only+Abstain | 0.438 | 0.538 | 0.427 |
| RAG+NLI | 0.522 | 0.578 | 0.225 |
| RAG+NLI+Abstain | 0.462 | 0.444 | 0.353 |

### Mistral

| Condition | F1-SUPPORTED | F1-REFUTED | F1-MISLEADING |
|---|---|---|---|
| Baseline | 0.829 | 0.554 | 0.495 |
| Baseline+Abstain | 0.860 | 0.588 | 0.581 |
| RAG-only | 0.738 | 0.542 | 0.329 |
| RAG-only+Abstain | 0.760 | 0.451 | 0.667 |
| RAG+NLI | 0.722 | 0.603 | 0.257 |
| RAG+NLI+Abstain | 0.753 | 0.442 | 0.602 |