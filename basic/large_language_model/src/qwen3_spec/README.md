# Qwen3 Speculative Decoding Implementation

This implementation provides speculative decoding for Qwen3 models using PyTorch. Speculative decoding is an inference acceleration technique that uses a smaller "draft" model to predict multiple tokens in parallel, which are then verified by the larger "target" model.

## Features

- GPU acceleration support
- Token verification with proper rejection handling
- Fallback mechanism for rejected tokens
- Boundary condition handling
- Clean, well-documented code following PyTorch best practices

## Components

- `speculative_decoder.py`: Core speculative decoding logic
- `qwen3_speculative_example.py`: Example usage script
- `utils.py`: Helper utilities

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers library
- Accelerate (optional, for better GPU utilization)

## Usage

```python
from speculative_decoder import SpeculativeDecoder
from transformers import AutoTokenizer

# Load models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-7B")  # or your preferred Qwen3 variant
target_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-7B", torch_dtype=torch.float16)
draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.8B", torch_dtype=torch.float16)  # smaller variant

# Initialize speculative decoder
decoder = SpeculativeDecoder(target_model, draft_model, tokenizer)

# Generate text
prompt = "Once upon a time"
output = decoder.generate(prompt, max_new_tokens=100, num_draft_tokens=4)
print(output)
```