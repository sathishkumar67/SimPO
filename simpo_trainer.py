from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from schedulefree.adamw_schedulefree import AdamWScheduleFree
import lightning as L
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger


@dataclass
class SimpoConfig:
    seq_length: int = 1024
    batch_size: int = 4
    num_workers: int = 4
    tokenizer: str = "gpt2"
    model: str = "gpt2"
    pad_token: str = "<pad>"
    padding_strategy: str = "max_length"
    dataloader_is_shuffle: bool = True
    beta: float = 2.0 # [2.0, 2.5]
    gamma: float = 0.5 # [0.3, 0.5, 1.0, 1.2, 1.4, 1.6]
    learning_rate: float = 3e-4

config = SimpoConfig()

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
tokenizer.pad_token = config.pad_token

ds = load_dataset("Dahoas/rm_instruct_helpful_preferences", split="train")

df = ds.to_pandas()

df["chosen_response"] = df["prompt"] + df["chosen"]
df["rejected_response"] = df["prompt"] + df["rejected"]

df["chosen_response_ids"] = df["chosen_response"].apply(lambda x: tokenizer(x, max_length=config.seq_length, padding=config.padding_strategy, return_tensors="pt"))
df["rejected_response_ids"] = df["rejected_response"].apply(lambda x: tokenizer(x, max_length=config.seq_length, padding=config.padding_strategy, return_tensors="pt"))


class SimpoDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        chosen_response_ids = self.df["chosen_response_ids"][idx]["input_ids"].squeeze()
        rejected_response_ids = self.df["rejected_response_ids"][idx]["input_ids"].squeeze()
        chosen_attention_mask = self.df["chosen_response_ids"][idx]["attention_mask"].squeeze()
        rejected_attention_mask = self.df["rejected_response_ids"][idx]["attention_mask"].squeeze()
        return chosen_response_ids, chosen_attention_mask, rejected_response_ids, rejected_attention_mask

simpo_ds = SimpoDataset(df)
simpo_dl = DataLoader(simpo_ds, batch_size=config.batch_size, shuffle=config.dataloader_is_shuffle, num_workers=config.num_workers)


class SimpoTrainer(L.LightningModule):
    def __init__(self, model, beta, gamma):
        super().__init__()
        self.model = model
        self.beta = beta
        self.gamma = gamma
        self.optimizer = self.configure_optimizers()

    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.zero_grad()
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
    
    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.model.parameters(), lr=config.learning_rate)
        return optimizer

model = AutoModelForCausalLM.from_pretrained(config.model)

simpo_trainer = SimpoTrainer(model, config.beta, config.gamma)
trainer = Trainer(max_epochs=10, accelerator="cuda", logger=CSVLogger("logs", name="simpo"))
trainer.fit(simpo_trainer, simpo_dl)

    




























        # here the prompt and the chosen/rejected sequences are merged, need to solve this issue
        # need to add mask for chosen and rejected sequences
        # only need to calculate the log probabilities for chosen sequences, rejected sequences not prompt


        # optimized code for mathematical stability 
        # import torch.nn.functional as F

        # # Forward pass
        # chosen_logits = self.model(chosen_ids)["logits"]
        # rejected_logits = self.model(rejected_ids)["logits"]

        # # Compute log probabilities directly
        # chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        # rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)

        # # Apply mask before summing log probabilities across the sequence
        # chosen_mask = chosen_ids.ne(pad_token_id)  # Assuming pad_token_id is defined
        # rejected_mask = rejected_ids.ne(pad_token_id)

        # # Sum masked log probabilities for chosen and rejected sequences
        # sum_chosen_log_probs = (chosen_log_probs * chosen_mask).sum(dim=1)
        # sum_rejected_log_probs = (rejected_log_probs * rejected_mask).sum(dim=1)

        # # Compute rewards
        # chosen_reward = self.beta / seq_length * sum_chosen_log_probs 
        # rejected_reward = self.beta / seq_length * sum_rejected_log_probs 

        # # Calculate the loss using a numerically stable log-sigmoid trick
        # reward_diff = chosen_reward - rejected_reward - self.gamma
        # loss = F.softplus(-reward_diff).mean()

        # return 
        
    # this is the code from gemini, need to modify it to fit the new code
    #     def simpo_loss(
    #     self,
    #     policy_chosen_logps: torch.FloatTensor,
    #     policy_rejected_logps: torch.FloatTensor,
    # ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    #     """Compute the SimPO loss for a batch of policy model log probabilities.

    #     Args:
    #         policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
    #         policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

    #     Returns:
    #         A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
    #         The losses tensor contains the SimPO loss for each example in the batch.
    #         The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    #     """
    #     pi_logratios = policy_chosen_logps - policy_rejected_logps
    #     pi_logratios = pi_logratios.to(self.accelerator.device)
    #     logits = pi_logratios - self.gamma_beta_ratio

    #     if self.loss_type == "sigmoid":
    #         losses = (
    #             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
    #             - F.logsigmoid(-self.beta * logits) * self.label_smoothing
    #         )
    #     elif self.loss_type == "hinge":
    #         losses = torch.relu(1 - self.beta * logits)
    #     else:
    #         raise ValueError(
    #             f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
    #         )

    #     chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
    #     rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()

    #     return losses, chosen_rewards, rejected_rewards