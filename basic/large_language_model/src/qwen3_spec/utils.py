"""
Utility functions for speculative decoding implementation.
"""

import torch
import warnings
from typing import Optional, Tuple, List


def setup_device() -> torch.device:
    """Set up the appropriate device (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def prepare_inputs_for_model(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple] = None,
    use_cache: bool = True
) -> dict:
    """
    Prepare input dictionary for model forward pass.
    
    Args:
        input_ids: Token IDs tensor
        attention_mask: Attention mask tensor (optional)
        past_key_values: Cached key-value pairs from previous inference steps (optional)
        use_cache: Whether to use KV caching
        
    Returns:
        Dictionary of inputs ready for model forward pass
    """
    inputs = {"input_ids": input_ids}
    
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask
        
    if past_key_values is not None:
        inputs["past_key_values"] = past_key_values
        
    inputs["use_cache"] = use_cache
    
    return inputs


def sample_from_logits(
    logits: torch.Tensor, 
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    Sample tokens from logits with optional temperature scaling and top-k/top-p filtering.
    
    Args:
        logits: Raw model logits
        temperature: Temperature for sampling (default: 1.0)
        top_k: Top-k filtering (optional)
        top_p: Top-p (nucleus) filtering (optional)
        
    Returns:
        Sampled token IDs
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
    
    # Apply top-p filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, 
            index=sorted_indices, 
            src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')
    
    # Sample from the filtered distribution
    probs = torch.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    
    return next_tokens.squeeze(-1)


def validate_draft_tokens(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    gamma: int,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, int]:
    """
    Validate draft tokens using the target model's logits.
    
    Implements the rejection sampling algorithm from the speculative decoding paper.
    
    Args:
        target_logits: Logits from the target model for the draft tokens
        draft_logits: Logits from the draft model for the draft tokens  
        draft_tokens: The draft tokens to validate
        gamma: Number of draft tokens generated
        temperature: Sampling temperature
        
    Returns:
        Tuple of (accepted_tokens, num_accepted) where accepted_tokens includes
        all accepted tokens plus potentially one additional token sampled from
        the residual distribution
    """
    batch_size = draft_tokens.shape[0]
    
    # Initialize acceptance mask
    accepted = torch.ones(batch_size, gamma + 1, dtype=torch.bool, device=draft_tokens.device)
    
    # Calculate acceptance probabilities for each draft token
    for i in range(gamma):
        # Get probabilities from both models
        target_probs = torch.softmax(target_logits[:, i] / temperature, dim=-1)
        draft_probs = torch.softmax(draft_logits[:, i] / temperature, dim=-1)
        
        # Get the probability of the draft token under both distributions
        draft_token_idx = draft_tokens[:, i]
        target_prob = target_probs.gather(-1, draft_token_idx.unsqueeze(-1)).squeeze(-1)
        draft_prob = draft_probs.gather(-1, draft_token_idx.unsqueeze(-1)).squeeze(-1)
        
        # Calculate acceptance probability
        acceptance_prob = torch.minimum(
            torch.ones_like(target_prob),
            target_prob / (draft_prob + 1e-8)  # Add small epsilon to avoid division by zero
        )
        
        # Accept or reject based on uniform random sample
        rand = torch.rand_like(acceptance_prob)
        accept_token = rand < acceptance_prob
        
        # Mark rejected tokens and all subsequent tokens as rejected
        rejected_mask = ~accept_token
        if rejected_mask.any():
            accepted[rejected_mask, i:] = False
            break
    
    # Count accepted tokens per batch
    num_accepted_per_batch = accepted.sum(dim=-1)
    
    # Handle the case where all draft tokens are accepted
    all_accepted = (num_accepted_per_batch == gamma + 1)
    
    # For sequences where all draft tokens were accepted, we need to sample one more token
    if all_accepted.any():
        # Sample from the residual distribution for fully accepted sequences
        residual_logits = target_logits[all_accepted, -1]
        additional_tokens = sample_from_logits(residual_logits, temperature=temperature)
        final_tokens = torch.cat([draft_tokens[all_accepted], additional_tokens.unsqueeze(-1)], dim=-1)
    else:
        # Truncate to only accepted tokens
        max_accepted = num_accepted_per_batch.max().item()
        final_tokens = draft_tokens[:, :max_accepted]
    
    # Return the maximum number of accepted tokens across the batch
    num_accepted = num_accepted_per_batch.max().item()
    
    return final_tokens, num_accepted