from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnlikelihoodTrainer(Trainer):
    def __init__(self, *args, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        # Initialize CrossEntropyLoss with the padding token (-100) ignored.
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def compute_prefix_unlikelihood_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch, seq_len, vocab_size = logits.size()  # [B, T, V]

        # Convert labels into one-hot vectors.
        # Clamp PAD token (-100) to 0 before one-hot encoding, then zero out the one-hot entries for PAD positions.
        lab = labels.clamp(min=0)              # Converts values from [-100, V-1] to [0, V-1].
        lab_oh = F.one_hot(lab, num_classes=vocab_size)  # Shape: [B, T, V]
        pad_mask = (labels == -100).unsqueeze(-1)        # Shape: [B, T, 1]
        lab_oh = lab_oh.masked_fill(pad_mask, 0)           # Zero out one-hot vectors at PAD positions.

        # Compute a cumulative sum along the time dimension.
        #    This counts how many times each token has appeared up to each timestep.
        prefix_counts = torch.cumsum(lab_oh, dim=1)  # Shape: [B, T, V]

        #Build prefix mask
        prefix_mask = torch.zeros_like(prefix_counts).bool()  
        prefix_mask[:, 1:, :] = prefix_counts[:, :-1, :].bool()  # Shift prefix info by one timestep.

        # 4. Compute token probabilities and apply the penalty
        probs = torch.softmax(logits, dim=-1)  # Shape: [B, T, V]
        prefix_probs = probs.masked_select(prefix_mask)  # Flatten

        # Calculate the Unlikelihood loss: -log(1 - p) 
        ul_vals = -torch.log(1.0 - prefix_probs + 1e-8)

        # Normalize the loss 
        ul_loss = ul_vals.mean()
        return ul_loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        # Compute Cross-Entropy Loss.
        ce = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        # Compute Unlikelihood Loss
        ul = self.compute_prefix_unlikelihood_loss(logits, labels)
        # Combine the two losses.
        loss = ce + self.alpha * ul
        return (loss, outputs) if return_outputs else loss
