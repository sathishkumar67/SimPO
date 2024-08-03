import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import lightning as L

PREFERED_COLUMNS = ["prompt", "chosen", "rejected"]

def prepare_dataset(dataset_name, tokenizer,split="train"): 
    # load the dataset
    dataset = load_dataset(dataset_name, split=split)
    # converting the dataset to pandas
    df = dataset.to_pandas()
    for column in df.columns:
        if column not in PREFERED_COLUMNS:
            raise ValueError(f"Only prompt, chosen, rejected columns are supported!")

    # tokenize the dataset
    df["input_ids"] = df["prompt"].apply(lambda x: tokenizer(x).input_ids)
    df["chosen_ids"] = df["chosen"].apply(lambda x: tokenizer(x).input_ids)
    df["rejected_ids"] = df["rejected"].apply(lambda x: tokenizer(x).input_ids)
    # count the number of tokens in each sequence of input_ids
    df["input_ids_count"] = df["input_ids"].apply(lambda x: len(x))
    return df



class SimpoTrainer(L.LightningModule):
    def __init__(self, model, beta, if_schedulefree_optimizer=True):
        super().__init__()
        self.model = model
        self.beta = beta
        self.if_schedulefree_optimizer = if_schedulefree_optimizer
        self.optimizer = self.configure_optimizers()

    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.zero_grad()
        if self.if_schedulefree_optimizer:
            optimizer.train()

        chosen_ids, rejected_ids, input_ids_count = batch

        # get logits for chosen and rejected sequences
        chosen_ids_logits = self.model(chosen_ids)
        rejected_ids_logits = self.model(rejected_ids)

        # get log probabilities for chosen and rejected sequences
        chosen_ids_log_probs = F.log_softmax(chosen_ids_logits, dim=-1)
        rejected_ids_log_probs = F.log_softmax(rejected_ids_logits, dim=-1)

        # only get the log probabilities from input_ids_count
        chosen_ids_log_probs = chosen_ids_log_probs[:, :, input_ids_count-1:-1]
        rejected_ids_log_probs = rejected_ids_log_probs[:, :, input_ids_count-1:-1]



def r_simpo_loss(model, input_ids, target_ids, beta):
    # input_ids: the input sequence (x)
    # target_ids: the target sequence (y)
    # model: the language model (like GPT) to compute log probabilities
    # beta: scaling factor

    # Forward pass through the model to get log probabilities
    output = model(input_ids=input_ids, labels=target_ids)
    log_probs = F.log_softmax(output.logits, dim=-1)
    
    # Gather the log probabilities corresponding to the target sequence
    # target_ids has shape (batch_size, sequence_length)
    # log_probs has shape (batch_size, sequence_length, vocab_size)
    batch_size, seq_length = target_ids.shape
    log_probs_target = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    
    # Sum log probabilities for the entire sequence for each example in the batch
    sum_log_probs = log_probs_target.sum(dim=1)
    
    # Normalize by sequence length (|y|)
    avg_log_probs = sum_log_probs / seq_length
    
    # Scale by beta
    loss = beta * avg_log_probs.mean()
    
    return loss

# Example usage:
# Assuming we have a model, input_ids, target_ids, and a beta value
# loss = r_simpo_loss(model, input_ids, target_ids, beta)