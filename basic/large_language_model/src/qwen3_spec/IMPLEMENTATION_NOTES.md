# Implementation Notes: Qwen3 Speculative Decoding

## Overview

This implementation provides speculative decoding for Qwen3 models following the algorithm described in "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023).

## Key Components

### 1. Core Algorithm

The speculative decoding algorithm works as follows:

1. **Draft Generation**: Use a smaller "draft" model to generate `γ` (gamma) candidate tokens autoregressively
2. **Parallel Verification**: Run the larger "target" model on the entire sequence (original + draft tokens) in parallel
3. **Token Validation**: Compare the target and draft model probabilities for each draft token
4. **Acceptance/Rejection**: Accept tokens based on rejection sampling, with fallback to direct sampling from the target model

### 2. Token Validation Logic

The validation uses the following acceptance probability for each draft token `x_i`:

```
P(accept x_i) = min(1, P_target(x_i) / P_draft(x_i))
```

Where:
- `P_target(x_i)` is the probability of token `x_i` under the target model
- `P_draft(x_i)` is the probability of token `x_i` under the draft model

If all `γ` draft tokens are accepted, an additional token is sampled from the residual distribution of the target model.

### 3. Boundary Condition Handling

The implementation handles several edge cases:

- **Empty acceptance**: If no draft tokens are accepted, fall back to sampling one token directly from the target model
- **EOS handling**: Stop generation when end-of-sequence tokens are encountered
- **Maximum length**: Respect the `max_new_tokens` limit even when generating draft tokens
- **Batch processing**: Handle multiple sequences in a batch with potentially different acceptance counts

### 4. GPU Acceleration

- Automatic device detection (CUDA, MPS, or CPU)
- Mixed precision support (float16 for GPU efficiency)
- KV caching for both draft and target models to minimize redundant computation
- Memory-efficient tensor operations

### 5. PyTorch Best Practices

- Models set to evaluation mode (`model.eval()`)
- Gradient computation disabled (`torch.set_grad_enabled(False)`)
- Proper tensor device management
- Efficient memory usage through caching
- Type hints and comprehensive documentation

## Usage Considerations

### Model Requirements

- Both target and draft models should be compatible (same tokenizer, similar architecture)
- Draft model should be significantly smaller/faster than target model for speedup
- Models should support the same vocabulary and tokenization scheme

### Performance Tuning

- **Gamma (num_draft_tokens)**: Higher values increase potential speedup but also rejection risk
- **Temperature**: Affects both draft generation quality and acceptance rates
- **Top-k/Top-p**: Can be used for controlled generation while maintaining speed benefits

### Limitations

- The current implementation assumes both models use the same tokenizer
- KV cache management could be further optimized for very long sequences
- The validation logic currently processes all sequences in a batch uniformly

## Future Improvements

1. **Adaptive gamma**: Dynamically adjust the number of draft tokens based on acceptance rates
2. **Better cache management**: More sophisticated KV cache handling for variable-length acceptance
3. **Multi-draft models**: Support ensembles of draft models for better prediction quality
4. **Memory optimization**: Further reduce memory footprint for very large models

## References

- Chen, X., Shi, C., & Zhang, C. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. arXiv preprint arXiv:2302.01318.
- Lei, W., et al. (2023). Qwen Technical Report. arXiv preprint arXiv:2309.16609.