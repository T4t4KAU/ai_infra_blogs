"""
Example script demonstrating speculative decoding with Qwen3 models.

This script shows how to use the speculative decoder with Qwen3 models.
Note: You'll need to have the appropriate Qwen3 model weights available
through HuggingFace Transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from speculative_decoder import SpeculativeDecoder


def main():
    """Main example function."""
    # Model names - adjust these based on available Qwen3 models
    # For demonstration, we'll use placeholder model names
    TARGET_MODEL_NAME = "/models/Qwen3-1.7B"  # Larger target model
    DRAFT_MODEL_NAME = "/models/Qwen3-0.6B"  # Smaller draft model
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load {TARGET_MODEL_NAME}. Using default Qwen tokenizer.")
        return
    
    print("Loading target model...")
    try:
        target_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"  # Automatically handle device placement
        )
    except Exception as e:
        print(f"Could not load target model {TARGET_MODEL_NAME}: {e}")
        return
    
    print("Loading draft model...")
    try:
        draft_model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
    except Exception as e:
        print(f"Could not load draft model {DRAFT_MODEL_NAME}: {e}")
        # Try to use the same model for both (not ideal but will work for demonstration)
        print("Using target model as draft model for demonstration...")
        draft_model = target_model
    
    # Initialize speculative decoder
    print("Initializing speculative decoder...")
    decoder = SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer
    )
    
    # Example prompts
    prompts = [
        "Once upon a time, in a land far away,",
        "The future of artificial intelligence is",
        "Explain quantum computing in simple terms:"
    ]
    
    print("\nGenerating with speculative decoding...")
    print("=" * 50)
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        print("-" * 30)
        
        try:
            # Generate with speculative decoding
            output = decoder.generate(
                prompt,
                max_new_tokens=50,
                num_draft_tokens=4,
                temperature=0.7,
                top_p=0.9
            )
            print(f"Output: {output}")
            
        except Exception as e:
            print(f"Error generating output: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("Speculative decoding example completed!")


def benchmark_comparison():
    """Compare speculative decoding vs regular decoding speed."""
    import time
    
    # This would require implementing regular decoding for comparison
    # For now, just show the structure
    print("Benchmark comparison would go here...")
    print("In practice, you would:")
    print("1. Time regular autoregressive generation")
    print("2. Time speculative decoding generation")  
    print("3. Compare tokens/second and output quality")


if __name__ == "__main__":
    main()
    
    # Uncomment to run benchmark (requires additional implementation)
    # benchmark_comparison()