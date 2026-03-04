# MoE-BERT: Mixture of Experts BERT — Proof of Concept

A proof-of-concept that replaces the standard feedforward layers in a BERT-style encoder with a Mixture-of-Experts (MoE) layer.

---

## Our Approach

We started from a ModernBERT-style encoder baseline and made the following changes:

**1. Replaced the FFN with a MoE layer (`FFNwMoE` in `model.py`)**

Instead of a single dense feedforward network, every transformer block now contains 16 expert networks. Each expert is a small GeGLU FFN. For every input token, only 2 experts are activated — 1 routed (selected dynamically per token) and 1 shared (always active, inspired by DeepSeekMoE). Expert weights are stored as stacked tensors (`w1_stacked`, `w2_stacked`, `w3_stacked`) for efficient batched dispatch.

**2. Loss-free load balancing (`use_lossfreebalance`)**

Traditional MoE uses an auxiliary loss to prevent all tokens from routing to the same expert. We replaced this with DeepSeek's loss-free balancing strategy — learnable per-expert bias values that are updated based on historical usage. This removes interference gradients during backpropagation and improved our routing stability significantly.

**3. Unpadding strategy**

We unpad the input once before the embedding layer, run all transformer blocks on the compact `(total_nnz, dim)` tensor, and repad only once at the end. This avoids wasted compute on padding tokens throughout the entire forward pass, not just in attention.

**4. Full HuggingFace compatibility (`CustomTransformerConfig`)**

The model is wrapped as a proper `PreTrainedModel` with `AutoConfig`, `AutoModelForMaskedLM`, and `AutoModelForSequenceClassification` support. This makes it easy to load, fine-tune, and evaluate using standard HuggingFace tools like Trainer and GLUE benchmarks.

**5. Efficiency additions**

- FlashAttention 2 for memory-efficient attention
- Liger Kernel's `LigerLayerNorm` as a drop-in replacement for PyTorch LayerNorm (~5% throughput improvement)
- `torch.compile()` on all compatible modules
- bfloat16 mixed precision training

---

## Training Setup

| Config | Value |
|---|---|
| Dataset | C4 (Colossal Cleaned Common Crawl) |
| Steps | 7,000 |
| Sequence length | 128 |
| Batch size | 4,096 |
| Learning rate | 4e-4 |
| MLM masking ratio | 30% |
| Precision | bfloat16 |
| Hardware | NVIDIA H200 |
| Training time | ~3.89 hours |

---

## Project Structure

```
MoE_BERT/
├── code/               # Model architecture and training scripts
├── data/               # Data preparation and tokenization
├── .gitattributes
├── .gitignore
├── README.md
├── requirements.txt    # Python dependencies
├── apt.txt             # System dependencies
├── variables.env       # Environment variables (do not share secrets)
├── preBuild.bash       # Setup script
└── postBuild.bash      # Post-build script
```

---

## Limitations

- Trained on short sequences (128 tokens) due to compute budget — MoE models are expected to benefit more at larger scale
- Only evaluated on GLUE — not tested on retrieval or semantic search tasks
- Hyperparameters were not fully tuned for MoE (dense model hyperparameters were reused)

---

## License

MIT License — feel free to use, modify, and build on this work with credit.
