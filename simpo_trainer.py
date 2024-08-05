import torch
import torch.nn.functional as F
import lightning as L

class SimpoTrainer(L.LightningModule):
    def __init__(self, model, beta, gamma, if_schedulefree_optimizer=True):
        super().__init__()
        self.model = model
        self.beta = beta
        self.gamma = gamma
        self.if_schedulefree_optimizer = if_schedulefree_optimizer
        self.optimizer = self.configure_optimizers()

    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.zero_grad()
        if self.if_schedulefree_optimizer:
            optimizer.train()

        chosen_ids, rejected_ids = batch
        _, seq_length = chosen_ids.shape

        # get logits for chosen and rejected sequences
        chosen_ids_logits = self.model(chosen_ids)
        rejected_ids_logits = self.model(rejected_ids)

        # get log probabilities for chosen and rejected sequences
        chosen_ids_log_probs = F.log_softmax(chosen_ids_logits, dim=-1)
        rejected_ids_log_probs = F.log_softmax(rejected_ids_logits, dim=-1) 

        # sum the log probabilities for chosen and rejected sequences
        sum_chosen_ids_log_probs = chosen_ids_log_probs.sum(dim=1)
        sum_rejected_ids_log_probs = rejected_ids_log_probs.sum(dim=1)

        # multiply by beta and normalize by sequence length
        chosen_reward = self.beta / seq_length * sum_chosen_ids_log_probs 
        rejected_reward = self.beta / seq_length * sum_rejected_ids_log_probs 
        
        # calculate the loss
        loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward - self.gamma)).mean()

        return loss
    

        # here the prompt and the chosen/rejected sequences are merged, need to solve this issue
        # need to add mask for chosen and rejected sequences
        # only need to calculate the log probabilities for chosen sequences, rejected sequences not prompt


        # optimized code 
        import torch.nn.functional as F

        # Forward pass
        chosen_logits = self.model(chosen_ids)["logits"]
        rejected_logits = self.model(rejected_ids)["logits"]

        # Compute log probabilities directly
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)

        # Apply mask before summing log probabilities across the sequence
        chosen_mask = chosen_ids.ne(pad_token_id)  # Assuming pad_token_id is defined
        rejected_mask = rejected_ids.ne(pad_token_id)

        # Sum masked log probabilities for chosen and rejected sequences
        sum_chosen_log_probs = (chosen_log_probs * chosen_mask).sum(dim=1)
        sum_rejected_log_probs = (rejected_log_probs * rejected_mask).sum(dim=1)

        # Compute rewards
        chosen_reward = self.beta / seq_length * sum_chosen_log_probs 
        rejected_reward = self.beta / seq_length * sum_rejected_log_probs 

        # Calculate the loss using a numerically stable log-sigmoid trick
        reward_diff = chosen_reward - rejected_reward - self.gamma
        loss = F.softplus(-reward_diff).mean()

        return loss