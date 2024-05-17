import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import random
import numpy as np
from tqdm import tqdm
from transformers import LongformerConfig, LongformerSelfAttention

# Global variables
PROGRAM_NAME = "panGPT"
VERSION = "0.10a"
AUTHOR = "James McInerney"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("training.log"), 
        logging.StreamHandler()
    ],
)

def print_banner():
    """
    Print the program banner with information about the program name, version, and author.

    This function prints the banner with the program name, version, and author information,
    along with additional disclaimer and license details.
    """
    border_symbol = "="
    padding_outer = 5
    padding_inner = 3
    full_program_name = f"{PROGRAM_NAME} v{VERSION}"
    line_length = len(full_program_name) + 2 * padding_inner
    border_line = border_symbol * (line_length + 2 * padding_outer)

    print(border_line)
    print(f"{border_symbol * padding_outer}{' ' * padding_inner}{full_program_name}{' ' * padding_inner}{border_symbol * padding_outer}")
    print(border_line)
    print(f"Developed by: {AUTHOR}")
    print("Website: http://mcinerneylab.com/")
    print("DISCLAIMER: This code is provided 'AS IS' without any warranties of any kind, including,")
    print("but not limited to, its fitness for a particular purpose. The author disclaims all ")
    print("liability for any damages, direct, indirect, tangential, incidental or consequential, ")
    print("resulting from the use of this code.")
    print("Licensed under the GNU General Public License v3.0")
    print("Full license at: https://www.gnu.org/licenses/gpl-3.0.en.html")
    print(border_line)

print_banner()

# Command line argument parsing
def parse_args():
    """
    Parse command-line arguments.

    This function parses the command-line arguments provided by the user and returns
    a Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a transformer model on (pan)'omic data.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "longformer"], help="Type of model to use: 'transformer' or 'longformer'")
    parser.add_argument("--longformer_attention_window", type=int, default=512, help="Attention window size in the Longformer model (default: 512)")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--model_dropout_rate", type=float, default=0.2, help="Dropout rate for the model")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5, help="Factor by which the learning rate will be reduced by the learning rate scheduler")
    parser.add_argument("--lr_patience", type=int, default=10, help="Patience for learning rate reduction")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--early_stop_patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=0.01, help="Minimum delta for early stopping")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max_vocab_size", type=int, default=70000, help="Maximum vocabulary size. Tokens beyond this size will be mapped to <UNK>.")
    parser.add_argument("--model_save_path", type=str, default="./model_checkpoint.pth", help="Path to save the model checkpoint")
    parser.add_argument("--tokenizer_file", type=str, default="./pangenome_gpt_tokenizer.json", help="Filename for saving and loading the tokenizer")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of the dataset to include in the training set")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of the dataset to include in the validation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pe_max_len", type=int, default=5000, help="Maximum length for positional encoding")
    parser.add_argument("--pe_dropout_rate", type=float, default=0.1, help="Dropout rate for positional encoding")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save TensorBoard logs")
    
    args = parser.parse_args()

    # Ensure pe_max_len is greater than or equal to max_seq_length
    if args.pe_max_len < args.max_seq_length:
        raise ValueError(f"Error: pe_max_len ({args.pe_max_len}) must be greater than or equal to max_seq_length ({args.max_seq_length}).")

    if args.model_type == "longformer":
        # Ensure max_seq_length is greater than or equal to longformer_attention_window
        args.max_seq_length = max(args.max_seq_length, args.longformer_attention_window)
        # Round down max_seq_length to the nearest multiple of longformer_attention_window
        args.max_seq_length = (args.max_seq_length // args.longformer_attention_window) * args.longformer_attention_window


    return args

args = parse_args()
params = vars(args)  # Convert the parsed arguments to a dictionary

input_file = args.input_file
model_type = args.model_type
longformer_attention_window=args.longformer_attention_window
embed_dim = args.embed_dim
num_heads = args.num_heads
num_layers = args.num_layers
max_seq_length = args.max_seq_length
batch_size = args.batch_size
model_dropout_rate = args.model_dropout_rate
learning_rate = args.learning_rate
lr_scheduler_factor = args.lr_scheduler_factor
weight_decay = args.weight_decay
lr_patience = args.lr_patience
early_stop_patience =args.early_stop_patience
min_delta = args.min_delta
epochs = args.epochs
max_vocab_size = args.max_vocab_size
model_save_path = args.model_save_path
tokenizer_file = args.tokenizer_file
train_size = args.train_size
val_size = args.val_size
seed = args.seed
pe_max_len = args.pe_max_len
pe_dropout_rate = args.pe_dropout_rate
log_dir = args.log_dir

# Check if max_seq_length is a multiple of longformer_attention_window when using Longformer
if model_type == "longformer" and max_seq_length % longformer_attention_window != 0:
    logging.info(f"Error: When using the Longformer model, the maximum sequence length (max_seq_length) must be a multiple of the attention window size (longformer_attention_window).")
    logging.info(f"Current values: max_seq_length = {max_seq_length}, longformer_attention_window = {longformer_attention_window}")
    logging.info("Please adjust these values and try again.")
    exit(1)


# Check whether certain files exist
if not os.path.isfile(input_file):
    print(f"Error: The specified input file '{input_file}' does not exist.")
    exit(1)
if model_save_path and not os.path.isdir(os.path.dirname(model_save_path)):
    print(f"Error: The directory for model save path '{model_save_path}' does not exist.")
    exit(1)

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
    - seed (int): The random seed value to set.

    This function sets the random seed for the random number generators in PyTorch,
    NumPy, and Python's built-in random module to ensure reproducibility of results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

def print_parameters_table(params: dict) -> None:
    """
    Prints a table of parameters and their settings.

    Args:
    - params (dict): Dictionary of parameters and their values.
    """
    try:
        max_param_length = max(len(param) for param in params)
        header = "-" * (max_param_length + 20)
        logging.info("\nParameters:")
        logging.info(header)
        logging.info("{:<{width}}: {}".format("Parameter", "Setting", width=max_param_length))
        logging.info(header)

        for param, value in params.items():
            if param == "longformer_attention_window" and params.get("model_type") != "longformer":
                continue  # Skip this parameter unless 'longformer' model is selected
            logging.info("{:<{width}}: {}".format(param, value, width=max_param_length))

        logging.info(header)

    except Exception as e:
        logging.error(f"Failed to log parameters: {e}")

print_parameters_table(params)

def load_dataset(input_file):
    """
    Load the dataset from the input file.

    Args:
    - input_file (str): Path to the input file containing the dataset.

    Returns:
    - list: List of strings, each representing a genome sequence.

    This function reads the contents of the input file, which contains genome sequences,
    and returns a list of genome sequences.
    """
    try:
        with open(input_file, "r") as file:
            genomes = [genome.strip() for genome in file.readlines()]
        return genomes
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit(1)

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Save the model checkpoint to a file.

    Args:
    - model: The PyTorch model to save.
    - optimizer: The optimizer state associated with the model.
    - epoch (int): The current epoch number.
    - loss: The loss value at the current epoch.
    - save_path (str): Path to save the model checkpoint file.

    This function saves the model checkpoint, including the model state dictionary,
    optimizer state dictionary, current epoch number, and loss value, to the specified file.
    """

    try:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, save_path)
    except IOError as e:
        print(f"Failed to save checkpoint to '{save_path}': {e}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load a model checkpoint from a file.

    Args:
    - model: The PyTorch model to load the checkpoint into.
    - optimizer: The optimizer associated with the model.
    - checkpoint_path (str): Path to the model checkpoint file.

    Returns:
    - tuple: A tuple containing the start epoch number and a boolean indicating
             whether the checkpoint was successfully loaded.

    This function loads a model checkpoint from the specified file into the given model
    and optimizer. It returns the start epoch number and a boolean indicating whether
    the checkpoint was successfully loaded.
    """

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting from scratch.")
        return 0, False
    try:
        checkpoint = torch.load(checkpoint_path)
        if not _is_compatible_checkpoint(model, checkpoint):
            print("Checkpoint incompatible. Starting from scratch.")
            return 0, False
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")
        return start_epoch, True
    except Exception as e:
        if "size mismatch" in str(e):
            error_msg = "Error: Checkpoint and current model do not match in size."
        else:
            error_msg = f"Error loading checkpoint from '{checkpoint_path}': {str(e)}"
        logging.error(error_msg)
        return 0, False

def _is_compatible_checkpoint(model, checkpoint):
    """
    Check if a model checkpoint is compatible with the current model.

    Args:
    - model: The PyTorch model.
    - checkpoint (dict): The model checkpoint.

    Returns:
    - bool: True if the checkpoint is compatible, False otherwise.

    This function checks if a model checkpoint is compatible with the current model
    by comparing the vocabulary sizes of the model and the checkpoint.
    """

    return model.vocab_size == checkpoint["model_state_dict"]["embed.weight"].size(0)

def train_model(train_loader, model, optimizer, criterion, device):
    """
    Train the transformer model on the training dataset.

    Args:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - model (nn.Module): Transformer model to train.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - criterion: Loss criterion for computing the loss.
    - device (torch.device): Device to perform computations on (CPU or GPU).

    Returns:
    - float: Average training loss.

    This function trains the transformer model on the training dataset for one epoch,
    computes the average training loss, and returns it.
    """

    model.train()  # Set the model to training mode
    total_train_loss = 0
    for i, (input_ids, labels) in enumerate(train_loader):  # Added enumeration for clarity
        input_ids, labels = input_ids.to(device), labels.to(device)  # Move data to the appropriate device
        optimizer.zero_grad()  # Clear gradients before calculating them
        outputs = model(input_ids)  # Generate predictions
        loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
        loss.backward()  # Compute gradient of the loss w.r.t. network parameters
        optimizer.step()  # Update parameters based on gradient

        total_train_loss += loss.item() * input_ids.size(0)  # Accumulate the loss
        # Update the progress bar
        train_loader.set_description(f"Epoch {epoch} - Training")
        train_loader.update(1)

    avg_train_loss = total_train_loss / train_dataset_size
    return avg_train_loss

def calculate_metrics(preds, labels):
    """
    Calculate evaluation metrics for model predictions.

    Args:
    - preds (Tensor): Predicted labels.
    - labels (Tensor): True labels.

    Returns:
    - tuple: Tuple containing evaluation metrics (accuracy, precision, recall, F1 score, Cohen's kappa).

    This function calculates various evaluation metrics (accuracy, precision, recall, F1 score, Cohen's kappa)
    based on the predicted labels and true labels, and returns them as a tuple.
    """

    preds = preds.view(-1)
    labels = labels.view(-1)
    if torch.unique(preds).size(0) == 1:
        warnings.warn("All predicted labels are the same. The model might not be learning properly.")
    accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average="weighted", zero_division=0)
    recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average="weighted", zero_division=0)
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="weighted", zero_division=0)
    kappa = cohen_kappa_score(labels.cpu().numpy(), preds.cpu().numpy())
    return accuracy, precision, recall, f1, kappa

def validate_model(val_loader, model, criterion, device, epoch=None):
    """
    Validate the transformer model on the validation dataset.

    Args:
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - model (nn.Module): Transformer model to validate.
    - criterion: Loss criterion for computing the loss.
    - device (torch.device): Device to perform computations on (CPU or GPU).

    Returns:
    - tuple: Tuple containing validation metrics (average validation loss, accuracy,
             precision, recall, F1 score, Cohen's kappa).

    This function validates the transformer model on the validation dataset,
    computes various validation metrics (average validation loss, accuracy, precision,
    recall, F1 score, Cohen's kappa), and returns them as a tuple.
    """

    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    total_accuracy = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for inputs, labels in val_loader:  # Correctly unpack the tuples returned by the DataLoader
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device

            outputs = model(inputs)  # Generate predictions from the model
            loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
            total_val_loss += loss.item() * inputs.size(0)  # Accumulate the loss

            preds = outputs.argmax(dim=-1)  # Get predicted classes
            correct = (preds == labels).sum().item()
            accuracy = correct / labels.numel()
            total_accuracy += accuracy * inputs.size(0)  # Accumulate the accuracy

            # Collect predictions and labels for calculating additional metrics
            preds_all.extend(preds.view(-1).tolist())
            labels_all.extend(labels.view(-1).tolist())
            # Update the progress bar
            if epoch is None:
                val_loader.set_description("Testing")
            else:
                val_loader.set_description(f"Epoch {epoch} - Validation")
            val_loader.update(1)

    # Calculate overall metrics from collected predictions and labels
    avg_val_loss = total_val_loss / val_dataset_size
    avg_val_accuracy = total_accuracy / val_dataset_size
    precision = precision_score(labels_all, preds_all, average='macro', zero_division=0)
    recall = recall_score(labels_all, preds_all, average='macro', zero_division=0)
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    kappa = cohen_kappa_score(labels_all, preds_all)

    return avg_val_loss, avg_val_accuracy, precision, recall, f1, kappa


genomes = load_dataset(input_file)
unique_tokens = set(token for genome in genomes for token in genome.split())
actual_vocab_size = len(unique_tokens)
vocab_size = min(actual_vocab_size, max_vocab_size)
num_sequences = len(genomes)
sequence_lengths = [len(genome.split()) for genome in genomes]
min_sequence_length = min(sequence_lengths)
max_sequence_length = max(sequence_lengths)
avg_sequence_length = sum(sequence_lengths) / num_sequences

logging.info(
    f"Dataset loaded: {num_sequences} sequences\n"
    f"Sequence lengths - Min: {min_sequence_length}, Max: {max_sequence_length}, Avg: {avg_sequence_length:.2f}"
)

tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
tokenizer.train_from_iterator(genomes, trainer)
tokenizer.save(tokenizer_file)
tokenizer = Tokenizer.from_file(tokenizer_file)
vocab_size = tokenizer.get_vocab_size()

if train_size + val_size > 1.0:
    raise ValueError("The sum of train_size and val_size must be less than or equal to 1.0")
if train_size + val_size == 1.0:
    train_genomes, val_genomes = train_test_split(genomes, train_size=train_size, random_state=seed)
    test_genomes = []
else:
    train_genomes, temp_genomes = train_test_split(genomes, train_size=train_size, random_state=seed)
    val_genomes, test_genomes = train_test_split(temp_genomes, test_size=1.0 - val_size / (1.0 - train_size), random_state=seed)

class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer input.

    This module adds positional encoding to the input embeddings to provide positional information
    to the transformer model.

    Args:
    - d_model (int): Dimension of the input embeddings.
    - max_len (int): Maximum length of the input sequence.
    - dropout (float): Dropout probability.

    Attributes:
    - pe (torch.Tensor): Positional encoding tensor.

    Methods:
    - forward(x): Forward pass of the positional encoding module.
    """

    def __init__(self, d_model, max_len, dropout=pe_dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class SimpleTransformerModel(nn.Module):
    """
    Simple transformer model for genomic data.

    This class defines a simple transformer model architecture for processing genomic data.

    Args:
    - vocab_size (int): Size of the vocabulary.
    - embed_dim (int): Dimension of the input embeddings.
    - num_heads (int): Number of attention heads.
    - num_layers (int): Number of transformer layers.
    - max_seq_length (int): Maximum sequence length.
    - dropout_rate (float): Dropout rate.
    - pe_max_len (int): Maximum length for positional encoding.
    - pe_dropout_rate (float): Dropout rate for positional encoding.

    Methods:
    - forward(x): Forward pass of the transformer model.

    Reference:
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., 
    Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I., 2017. 
    Attention is all you need. Advances in neural information processing systems, 30.
    """

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length, dropout_rate, pe_max_len, pe_dropout_rate):
        super(SimpleTransformerModel, self).__init__()
        self.pos_encoding = PositionalEncoding(embed_dim, pe_max_len, dropout=pe_dropout_rate)
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoding(x)  # Apply positional encoding after embedding
        x = self.transformer(x)
        return self.out(x)

class LongformerModel(nn.Module):
    """
    Longformer model for processing long sequences.

    This class defines a Longformer model that can handle long input sequences efficiently
    by using a combination of local and global attention mechanisms. It is based on the
    Longformer architecture introduced by Beltagy et al. (2020).

    Args:
    - vocab_size (int): Size of the vocabulary.
    - embed_dim (int): Dimension of the input embeddings.
    - num_heads (int): Number of attention heads.
    - num_layers (int): Number of Longformer layers.
    - max_seq_length (int): Maximum sequence length.
    - dropout_rate (float): Dropout rate for the model.
    - pe_max_len (int): Maximum length for positional encoding.
    - pe_dropout_rate (float): Dropout rate for positional encoding.
    - longformer_config (LongformerConfig): Configuration object for the Longformer model.

    Attributes:
    - pos_encoding (PositionalEncoding): Positional encoding module.
    - vocab_size (int): Size of the vocabulary.
    - embed (nn.Embedding): Embedding layer for input tokens.
    - longformer_layers (nn.ModuleList): List of Longformer self-attention layers.
    - out (nn.Linear): Output linear layer for token prediction.

    Methods:
        forward(x): Perform forward pass through the Longformer model.

    References:
        - Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer.
          arXiv preprint arXiv:2004.05150.
    """

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length,
                 dropout_rate, pe_max_len, pe_dropout_rate, longformer_config):
        super(LongformerModel, self).__init__()
        self.pos_encoding = PositionalEncoding(embed_dim, pe_max_len, dropout=pe_dropout_rate)
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.longformer_layers = nn.ModuleList([
            LongformerSelfAttention(longformer_config, layer_id=i)
            for i in range(num_layers)
        ])

        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoding(x)

        attention_mask = torch.ones(x.size()[:-1], dtype=torch.long, device=x.device)

        # Generate is_index_masked tensor
        is_index_masked = torch.zeros_like(attention_mask, dtype=torch.bool)

        for longformer_layer in self.longformer_layers:
            x = longformer_layer(x, attention_mask=attention_mask, is_index_masked=is_index_masked)[0]

        return self.out(x)




class GenomeDataset(torch.utils.data.Dataset):
    """
    Dataset class for genomic data.

    This class represents a dataset of genomic sequences for training, validation, or testing.

    Args:
    - texts (list): List of genomic sequences.
    - tokenizer (Tokenizer): Tokenizer for encoding genomic sequences.
    - max_length (int): Maximum length of the input sequence.

    Methods:
    - __len__(): Get the length of the dataset.
    - __getitem__(idx): Get an item from the dataset by index.
    """

    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.encode(text).ids

        # Ensure the sequence is not longer than max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        
        # Input is all but the last token
        input_ids = encoded[:-1]
        # Labels are all but the first token, shifted by one
        label_ids = encoded[1:]

        # Pad the sequence to the nearest multiple of the attention window size (for Longformer)
        if hasattr(self, 'attention_window'):
            seq_length = len(input_ids)
            padded_length = ((seq_length + self.attention_window - 1) // self.attention_window) * self.attention_window
            input_ids = input_ids + [self.tokenizer.token_to_id("[PAD]")] * (padded_length - seq_length)
            label_ids = label_ids + [self.tokenizer.token_to_id("[PAD]")] * (padded_length - seq_length)
        else:
            # Pad the sequence to max_length (for transformer)
            input_ids = input_ids + [self.tokenizer.token_to_id("[PAD]")] * (self.max_length - 1 - len(input_ids))
            label_ids = label_ids + [self.tokenizer.token_to_id("[PAD]")] * (self.max_length - 1 - len(label_ids))

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)


class EarlyStopping:
    """
    Early stopping handler for model training.

    This class implements early stopping functionality to stop model training
    when the validation loss stops improving.

    Args:
    - patience (int): Number of epochs to wait before stopping.
    - min_delta (float): Minimum change in loss to be considered an improvement.
    - verbose (bool): Whether to print early stopping messages.

    Methods:
    - __call__(val_loss): Check if early stopping criteria are met based on the validation loss.
    """

    def __init__(self, patience, min_delta=min_delta, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

if model_type == 'transformer':
    model = SimpleTransformerModel(vocab_size, embed_dim, num_heads, num_layers, max_seq_length, dropout_rate=model_dropout_rate, pe_max_len=pe_max_len, pe_dropout_rate=pe_dropout_rate)
elif model_type == 'longformer':
    attention_window = args.longformer_attention_window
    longformer_config = LongformerConfig(
        hidden_size=embed_dim,
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers,
        attention_window=[attention_window] * num_layers,
        intermediate_size=4 * embed_dim,
    )
    model = LongformerModel(vocab_size, embed_dim, num_heads, num_layers, max_seq_length, dropout_rate=model_dropout_rate, pe_max_len=pe_max_len, pe_dropout_rate=pe_dropout_rate, longformer_config=longformer_config)
else:
    raise ValueError(f"Invalid model type: {model_type}")

early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=min_delta, verbose=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}", flush=True)

# training dataset
train_dataset = GenomeDataset(train_genomes, tokenizer, max_seq_length)
if args.model_type == "longformer":
    train_dataset.attention_window = longformer_attention_window
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_dataset_size = len(train_loader.dataset)

# validation dataset
val_dataset = GenomeDataset(val_genomes, tokenizer, max_seq_length)
if args.model_type == "longformer":
    val_dataset.attention_window = longformer_attention_window
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_dataset_size = len(val_loader.dataset)

criterion = torch.nn.CrossEntropyLoss() # what are we trying to optimize?
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # How are we trying to optimizer it?
lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=lr_scheduler_factor, patience=lr_patience, verbose=True) # taking big, then small steps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Run on a GPU if one is available
logging.info(f"device = {device}")
model.to(device)

start_epoch, is_checkpoint_loaded = load_checkpoint(model, optimizer, model_save_path)

if is_checkpoint_loaded:
    logging.info("Continuing training from the loaded checkpoint.")
else:
    logging.info("Starting training from scratch.")
    start_epoch = 0

print(f"vocab_size: {vocab_size} | embed_dim: {embed_dim} | num_heads: {num_heads} | num_layers: {num_layers} | max_seq_length: {max_seq_length}", flush=True)
for epoch in range(start_epoch, epochs):
    writer = SummaryWriter(log_dir=log_dir)
    # Training model loop
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch} - Training", unit="batch")
    avg_train_loss = train_model(train_loader, model, optimizer, criterion, device)
    train_perplexity = torch.exp(torch.tensor(avg_train_loss))
    # Log training metrics
    logging.info(f'Epoch {epoch} - Training Loss: {avg_train_loss}, Perplexity: {train_perplexity}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Perplexity/train", train_perplexity, epoch)

    # Validate model loop
    val_loader = tqdm(val_loader, desc=f"Epoch {epoch} - Validation", unit="batch")
    avg_val_loss, val_accuracy, val_precision, val_recall, val_f1, val_kappa = validate_model(val_loader, model, criterion, device, epoch)
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))
    # Log validation metrics
    logging.info(f'Epoch {epoch} - Validation Loss: {avg_val_loss}, Perplexity: {val_perplexity}, Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}, Kappa: {val_kappa}')
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Perplexity/val", val_perplexity, epoch)
    writer.add_scalar("Accuracy/val", val_accuracy, epoch)
    writer.add_scalar("Precision/val", val_precision, epoch)
    writer.add_scalar("Recall/val", val_recall, epoch)
    writer.add_scalar("F1/val", val_f1, epoch)
    writer.add_scalar("Kappa/val", val_kappa, epoch)
    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

    lr_scheduler.step(avg_val_loss)
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered.", flush=True)
        break
    elif avg_val_loss <= early_stopping.best_loss:
        print("Saving model checkpoint.", flush=True)
        save_checkpoint(model, optimizer, epoch, avg_train_loss, model_save_path)

    gc.collect()
    writer.close()

if len(test_genomes) > 0:
    test_dataset = GenomeDataset(test_genomes, tokenizer, max_seq_length)
    if args.model_type == "longformer":
        test_dataset.attention_window = longformer_attention_window
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_dataset_size = len(test_loader.dataset)  # Store the size of the test dataset
    test_loader = tqdm(test_loader, desc="Testing", unit="batch")
    # Test Model Loop
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_kappa = validate_model(test_loader, model, criterion, device)
    test_perplexity = torch.exp(torch.tensor(test_loss))
    # Log test metrics
    logging.info(f'Test Loss: {test_loss}, Perplexity: {test_perplexity}, Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}, Kappa: {test_kappa}')
    # Create a new SummaryWriter instance for test metrics
    test_writer = SummaryWriter(log_dir=os.path.join(log_dir, "test"))
    test_writer.add_scalar("Loss/test", test_loss)
    test_writer.add_scalar("Perplexity/test", test_perplexity)
    test_writer.add_scalar("Accuracy/test", test_accuracy)
    test_writer.add_scalar("Precision/test", test_precision)
    test_writer.add_scalar("Recall/test", test_recall)
    test_writer.add_scalar("F1/test", test_f1)
    test_writer.add_scalar("Kappa/test", test_kappa)
    test_writer.close()

else:
    print("No test set available for evaluation.", flush=True)


