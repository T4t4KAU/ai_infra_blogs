"""
Speculative Decoding implementation for Qwen3 models.

This module implements the speculative decoding algorithm as described in
"Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from utils import (
    setup_device, 
    prepare_inputs_for_model, 
    sample_from_logits,
    validate_draft_tokens
)


class SpeculativeDecoder:
    """
    Speculative Decoder for accelerating LLM inference using a draft model.
    
    The decoder uses a smaller "draft" model to predict multiple tokens in parallel,
    which are then verified by the larger "target" model. This can significantly
    speed up inference while maintaining the output quality of the target model.
    """
    
    def __init__(
        self,
        target_model: PreTrainedModel,
        draft_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the speculative decoder.
        
        Args:
            target_model: The main (larger) model that produces high-quality outputs
            draft_model: The smaller model used for speculative token generation
            tokenizer: Tokenizer for the models
            device: Device to run inference on (auto-detected if None)
            dtype: Data type for model weights (default: float16 for GPU efficiency)
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        
        # Set up device and move models to appropriate device
        self.device = device if device is not None else setup_device()
        self.dtype = dtype
        
        self.target_model.to(self.device, dtype=self.dtype)
        self.draft_model.to(self.device, dtype=self.dtype)
        
        # Set models to evaluation mode
        self.target_model.eval()
        self.draft_model.eval()
        
        # Disable gradient computation for efficiency
        torch.set_grad_enabled(False)
        
    def _generate_draft_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gamma: int,
        past_key_values_draft: Optional[Tuple] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple, torch.Tensor]:
        """
        Generate gamma draft tokens using the draft model.
        
        Args:
            input_ids: Current sequence of token IDs
            attention_mask: Attention mask for the current sequence
            gamma: Number of draft tokens to generate
            past_key_values_draft: Cached KV pairs from previous draft model inference
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Top-p filtering parameter
            
        Returns:
            Tuple of (draft_tokens, draft_logits, new_past_key_values, draft_attention_mask)
        """
        batch_size, seq_len = input_ids.shape
        current_input_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        draft_tokens = []
        draft_logits_list = []
        past_key_values = past_key_values_draft
        
        # Generate gamma tokens autoregressively using the draft model
        for _ in range(gamma):
            # Prepare inputs for draft model
            draft_inputs = prepare_inputs_for_model(
                current_input_ids,
                current_attention_mask,
                past_key_values,
                use_cache=True
            )
            
            # Forward pass through draft model
            with torch.no_grad():
                draft_outputs = self.draft_model(**draft_inputs)
                logits = draft_outputs.logits[:, -1, :]  # Get logits for last token
                past_key_values = draft_outputs.past_key_values
                
            # Sample next token
            next_token = sample_from_logits(
                logits, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Store results
            draft_tokens.append(next_token)
            draft_logits_list.append(logits)
            
            # Update inputs for next iteration
            current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(-1)], dim=-1)
            current_attention_mask = torch.cat([
                current_attention_mask, 
                torch.ones(batch_size, 1, device=self.device, dtype=torch.long)
            ], dim=-1)
        
        # Stack results
        draft_tokens_tensor = torch.stack(draft_tokens, dim=1)  # [batch_size, gamma]
        draft_logits_tensor = torch.stack(draft_logits_list, dim=1)  # [batch_size, gamma, vocab_size]
        
        return draft_tokens_tensor, draft_logits_tensor, past_key_values, current_attention_mask
    
    def _verify_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        draft_tokens: torch.Tensor,
        past_key_values_target: Optional[Tuple] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple, int]:
        """
        Verify draft tokens using the target model.
        
        Args:
            input_ids: Original input sequence
            attention_mask: Attention mask for original sequence
            draft_tokens: Draft tokens to verify
            past_key_values_target: Cached KV pairs from previous target model inference
            temperature: Sampling temperature
            
        Returns:
            Tuple of (accepted_tokens, target_logits, new_past_key_values, num_accepted)
        """
        batch_size, seq_len = input_ids.shape
        gamma = draft_tokens.shape[1]
        
        # Create candidate sequence: original + draft tokens
        candidate_input_ids = torch.cat([input_ids, draft_tokens], dim=1)
        candidate_attention_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, gamma, device=self.device, dtype=torch.long)
        ], dim=1)
        
        # Get target model logits for the entire candidate sequence
        target_inputs = prepare_inputs_for_model(
            candidate_input_ids,
            candidate_attention_mask,
            past_key_values_target,
            use_cache=True
        )
        
        with torch.no_grad():
            target_outputs = self.target_model(**target_inputs)
            target_logits = target_outputs.logits[:, seq_len-1:seq_len+gamma, :]  # [batch_size, gamma+1, vocab_size]
            new_past_key_values = target_outputs.past_key_values
        
        # Get draft model logits for the draft tokens (needed for validation)
        # We need to run the draft model on the same candidate sequence
        draft_inputs = prepare_inputs_for_model(
            candidate_input_ids,
            candidate_attention_mask,
            None,  # Don't use cached KV for draft model in verification
            use_cache=False
        )
        
        with torch.no_grad():
            draft_outputs = self.draft_model(**draft_inputs)
            draft_verify_logits = draft_outputs.logits[:, seq_len-1:seq_len+gamma-1, :]  # [batch_size, gamma, vocab_size]
        
        # Validate tokens
        accepted_tokens, num_accepted = validate_draft_tokens(
            target_logits,
            draft_verify_logits,
            draft_tokens,
            gamma,
            temperature=temperature
        )
        
        return accepted_tokens, target_logits, new_past_key_values, num_accepted
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 50,
        num_draft_tokens: int = 4,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ) -> Union[str, List[str]]:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Input prompt(s) as string or list of strings
            max_new_tokens: Maximum number of new tokens to generate
            num_draft_tokens: Number of tokens to speculate per iteration (gamma)
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Top-p filtering parameter
            eos_token_id: End-of-sequence token ID (stops generation if encountered)
            pad_token_id: Padding token ID
            
        Returns:
            Generated text as string or list of strings
        """
        # Handle input types
        if isinstance(prompt, str):
            prompts = [prompt]
            return_single = True
        else:
            prompts = prompt
            return_single = False
        
        batch_size = len(prompts)
        
        # Set default token IDs if not provided
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or 0
        
        # Tokenize input prompts
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        
        # Initialize state
        generated_tokens = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        past_key_values_target = None
        past_key_values_draft = None
        
        tokens_generated = 0
        
        # Main generation loop
        while tokens_generated < max_new_tokens:
            # Check if we should stop early (all sequences hit EOS)
            if eos_token_id is not None:
                last_tokens = generated_tokens[:, -1]
                if (last_tokens == eos_token_id).all():
                    break
            
            # Calculate how many tokens we can still generate
            remaining_tokens = max_new_tokens - tokens_generated
            current_gamma = min(num_draft_tokens, remaining_tokens)
            
            if current_gamma <= 0:
                break
            
            # Generate draft tokens
            draft_tokens, draft_logits, past_key_values_draft, draft_attention_mask = self._generate_draft_tokens(
                generated_tokens,
                current_attention_mask,
                current_gamma,
                past_key_values_draft,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Verify draft tokens with target model
            accepted_tokens, target_logits, past_key_values_target, num_accepted = self._verify_tokens(
                generated_tokens,
                current_attention_mask,
                draft_tokens,
                past_key_values_target,
                temperature=temperature
            )
            
            # Handle the case where no tokens were accepted (should be rare but possible)
            if num_accepted == 0:
                # Sample one token directly from target model
                target_inputs = prepare_inputs_for_model(
                    generated_tokens,
                    current_attention_mask,
                    past_key_values_target,
                    use_cache=True
                )
                
                with torch.no_grad():
                    target_outputs = self.target_model(**target_inputs)
                    next_logits = target_outputs.logits[:, -1, :]
                    next_token = sample_from_logits(
                        next_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                    accepted_tokens = next_token.unsqueeze(-1)
                    num_accepted = 1
                    past_key_values_target = target_outputs.past_key_values
            
            # Update generated tokens and attention mask
            generated_tokens = torch.cat([generated_tokens, accepted_tokens], dim=1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones(batch_size, num_accepted, device=self.device, dtype=torch.long)
            ], dim=1)
            
            tokens_generated += num_accepted
            
            # Update past key values to only include the accepted tokens
            if past_key_values_target is not None:
                # Truncate past key values to match the accepted sequence length
                # This is a simplified approach - in practice, you might want to keep
                # the full cache and just track the valid length
                pass
        
        # Decode generated tokens back to text
        decoded_outputs = []
        for i in range(batch_size):
            # Find the actual end of sequence (before padding)
            seq = generated_tokens[i]
            if eos_token_id is not None:
                eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    seq = seq[:eos_positions[0] + 1]
            
            decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
            decoded_outputs.append(decoded)
        
        return decoded_outputs[0] if return_single else decoded_outputs