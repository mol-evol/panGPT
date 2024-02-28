import argparse
import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Global variables
PROGRAM_NAME = "panGPT"
VERSION = "0.02"
AUTHOR = "James McInerney"


def log_memory_usage():
    memory = psutil.virtual_memory()
    logging.info(f"Memory Usage: {memory.percent}% used, {memory.available / (1024**3):.2f} GB available")


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug_log.log"), logging.StreamHandler()],
)


def print_banner():
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
    print("DISCLAIMER: This code is provided 'AS IS' without any warranties of any kind, including, but not limited to, its fitness for a particular purpose. The author disclaims all liability for any damages, direct, indirect, tangential, incidental or consequential, resulting from the use of this code.")
    print(border_line)
    print("Licensed under the GNU General Public License v3.0")
    print("Full license at: https://www.gnu.org/licenses/gpl-3.0.en.html")
    print(border_line)

print_banner()

# Command line argument parsing
def parse_args():
    """
    Parse command-line arguments for the program.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.

    Defines and parses the following arguments: input_file, embed_dim, num_heads, num_layers,
    max_seq_length, batch_size, learning_rate, weight_decay, patience, min_delta, epochs,
    vocab_size, model_save_path and tokenizer_file.
    """
    parser = argparse.ArgumentParser(description="Train a transformer model on pangenomic presence-absence data.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input pangenome file")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=0.01, help="Minimum delta for early stopping")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--max_vocab_size", type=int, default=70000, help="Maximum vocabulary size. Tokens beyond this size will be mapped to <UNK>.")
    parser.add_argument("--model_save_path", type=str, default="model_checkpoint.pth", help="Path to save the model checkpoint")
    parser.add_argument("--tokenizer_file", type=str, default="pangenome_gpt_tokenizer.json", help="Filename for saving and loading the tokenizer")

    return parser.parse_args()


args = parse_args()
# Use args to access the command-line arguments
model_save_path = args.model_save_path
embed_dim = args.embed_dim
num_heads = args.num_heads
num_layers = args.num_layers
max_seq_length = args.max_seq_length
batch_size = args.batch_size
learning_rate = args.learning_rate
weight_decay = args.weight_decay
patience = args.patience
min_delta = args.min_delta
epochs = args.epochs


if not os.path.isfile(args.input_file):
    print(f"Error: The specified input file '{args.input_file}' does not exist.")
    exit(1)
if args.model_save_path and not os.path.isdir(os.path.dirname(args.model_save_path)):
    print(
        f"Error: The directory for model save path '{args.model_save_path}' does not exist."
    )
    exit(1)


def load_dataset(input_file):
    """
    Loads the dataset from the specified input file.

    Args:
        input_file (str): Path to the input file.

    Returns:
        list: A list of genomes read from the file.
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


def calculate_metrics(preds, labels):
    """
    Calculates accuracy, precision, recall, and F1 score from predictions and labels.

    Args:
        preds: Predicted labels (tensor)
        labels: True labels (tensor)

    Returns:
        Tuple: (accuracy, precision, recall, F1 score)
    """
    preds = preds.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")

    return accuracy, precision, recall, f1


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Saves the model's state dictionary, optimizer state, epoch, and loss as a checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current training epoch
        loss: Total training loss for the epoch
        save_path: Path to save the checkpoint file
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
    Loads a model checkpoint if it exists and is compatible.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        int: The starting epoch number for training.
        bool: Indicates if the checkpoint was loaded successfully.
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
        print(f"Error loading checkpoint from '{checkpoint_path}': {e}")
        return 0, False


def _is_compatible_checkpoint(model, checkpoint):
    """Checks for checkpoint compatibility based on vocab size and other criteria."""
    return model.vocab_size == checkpoint["model_state_dict"]["embed.weight"].size(0)


def train_model(train_loader, model, optimizer, criterion, device):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        input_ids = batch.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, model.vocab_size), input_ids.view(-1))
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * input_ids.size(0)

    return total_train_loss


def validate_model(val_loader, model, criterion, device):
    model.eval()
    total_val_loss = 0
    for batch in val_loader:
        input_ids = batch.to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, model.vocab_size), input_ids.view(-1))

        total_val_loss += loss.item() * input_ids.size(0)

    return total_val_loss


genomes = load_dataset(args.input_file)
unique_tokens = set(token for genome in genomes for token in genome.split())
actual_vocab_size = len(
    unique_tokens
)  # Size of the vocabulary in the pangenome dataset.
vocab_size = min(
    actual_vocab_size, args.max_vocab_size
)  # Set vocab_size to be the minimum of the command line argument or the token count in the pangenome dataset.

# Initialize and train the tokenizer using the 'tokenizers' library
tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordLevelTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size
)
tokenizer.train_from_iterator(genomes, trainer)
tokenizer.save(args.tokenizer_file)  # Save the trained tokenizer

# Load the trained tokenizer from the saved JSON file
tokenizer = Tokenizer.from_file(args.tokenizer_file)
vocab_size = tokenizer.get_vocab_size()  # Set vocab_size for the model

# Split the data into training and validation sets (80% training, 20% validation)
train_genomes, val_genomes = train_test_split(genomes, test_size=0.2, random_state=42)


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in the Transformer paper.

    Args:
        d_model (int): The dimension of the embeddings (also called the model dimension).
        dropout (float): Dropout rate.
        max_len (int): Maximum length of the input sequences.

    This module injects some information about the relative or absolute position of
    the tokens in the sequence to make use of the order of the sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class SimpleTransformerModel(nn.Module):
    """
    A simple Transformer model for sequence generation.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embedding layer.
        num_heads (int): Number of attention heads in the transformer.
        num_layers (int): Number of layers (stacks) in the transformer.
        max_seq_length (int): Maximum length of the input sequences.
        dropout_rate (float): Dropout rate in the transformer.

    The model consists of an embedding layer, positional encoding, and a transformer encoder.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        max_seq_length,
        dropout_rate=0.5,
    ):
        super(SimpleTransformerModel, self).__init__()
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout_rate)
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers
        )
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.out(x)


class GenomeDataset(Dataset):
    """
    GenomeDataset: A custom PyTorch Dataset for preprocessing genomic sequences.

    This class facilitates loading and preprocessing genomic sequences for training
    and evaluating deep learning models. It utilizes a provided tokenizer to
    convert text sequences into numerical representations suitable for model input.

    Args:
        texts (list): A list of text strings representing the genomic sequences.
        tokenizer (transformers.PreTrainedTokenizer): A tokenizer object for
            tokenizing the text sequences.
        max_length (int): The maximum allowed length for the processed sequences
            (after tokenization and padding).

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for
            tokenization.
        texts (list): The list of original text sequences.
        max_length (int): The maximum allowed length for the processed sequences.

    Methods:
        __len__() -> int: Returns the number of samples in the dataset.
        __getitem__(idx) -> torch.tensor: Returns a preprocessed genomic sequence
            (tensor) at the specified index.
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

        if len(encoded) > self.max_length:
            encoded = encoded[: self.max_length]

        padded = encoded + [self.tokenizer.token_to_id("[PAD]")] * (
            self.max_length - len(encoded)
        )
        return torch.tensor(padded)


class EarlyStopping:
    """
    EarlyStopping: A callback for stopping training early based on validation loss.

    This class implements the Early Stopping technique, which can help prevent overfitting
    during training of deep learning models. It monitors the validation loss and stops
    training if there is no significant improvement (decrease) in the loss for a
    specified number of epochs (`patience`).

    Args:
        patience (int, optional): The number of epochs with no improvement
            (decrease in validation loss) after which training will be stopped.
            Defaults to 5.
        min_delta (float, optional): The minimum change in validation loss to
            qualify as an improvement. Defaults to 0.
        verbose (bool, optional): Whether to print messages during training.
            Defaults to False.

    Attributes:
        patience (int): The patience value.
        min_delta (float): The minimum delta value.
        verbose (bool): The verbosity flag.
        counter (int): Internal counter for tracking epochs without improvement.
        best_loss (float): The best validation loss observed so far.
        early_stop (bool): Flag indicating whether early stopping has been triggered.

    Methods:
        __call__(self, val_loss): Called at the end of each epoch with the validation loss.
            Updates the internal state and returns `True` if early stopping is triggered,
            `False` otherwise.
    """

    def __init__(self, patience=5, min_delta=0, verbose=False):
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

model = SimpleTransformerModel(vocab_size, embed_dim, num_heads, num_layers, max_seq_length, dropout_rate=0.5)
early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}", flush=True)

# Create the datasets and data loaders for training and validation
train_dataset = GenomeDataset(train_genomes, tokenizer, max_seq_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = GenomeDataset(val_genomes, tokenizer, max_seq_length)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Loss and Optimizer: Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Initialize the learning rate scheduler
lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
print(f"device = {device}", flush=True)
model.to(device)  # Move the model to the GPU if available

start_epoch, is_checkpoint_loaded = load_checkpoint(model, optimizer, model_save_path)

if is_checkpoint_loaded:
    # If the checkpoint was loaded successfully
    print("Continuing training from the loaded checkpoint.", flush=True)
else:
    # If no checkpoint was loaded (either because it didn't exist or there was a mismatch)
    print("Starting training from scratch.", flush=True)
    start_epoch = 0  # Ensure training starts from the first epoch if no checkpoint was loaded

print(
    f"vocab_size: {vocab_size} | embed_dim: {embed_dim} | num_heads: {num_heads} | num_layers: {num_layers} | max_seq_length: {max_seq_length}",
    flush=True,
)
for epoch in range(start_epoch, epochs):
    train_loss = train_model(train_loader, model, optimizer, criterion, device)
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_perplexity = torch.exp(torch.tensor(avg_train_loss))

    # Print training metrics
    print(f"Epoch {epoch} - Training Loss: {avg_train_loss}, Perplexity: {train_perplexity}", flush=True)

    val_loss = validate_model(val_loader, model, criterion, device)
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))

    # Print validation metrics
    print(f"Epoch {epoch} - Validation Loss: {avg_val_loss}, Perplexity: {val_perplexity}", flush=True)

    lr_scheduler.step(avg_val_loss)

    # Early stopping check
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered.", flush=True)
        break
    elif avg_val_loss <= early_stopping.best_loss:
        print("Saving model checkpoint.", flush=True)
        save_checkpoint(model, optimizer, epoch, train_loss, model_save_path)

    # Optional garbage collection
    gc.collect()
