import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
import math
import torch.nn.functional as F


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

def print_banner():
    banner = '''
    **************************************************
    *                                                *
    *        Transformer Model Token Prediction      *
    *        panPrompt v0.01                         *
    *        author: James McInerney                 *
    *                                                *
    **************************************************
    '''
    print(banner)



def load_model(model_path, embed_dim, num_heads, num_layers, max_seq_length):
    # Infer the vocab size from the model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    vocab_size = checkpoint['model_state_dict']['embed.weight'].size(0)
    model = SimpleTransformerModel(vocab_size, embed_dim, num_heads, num_layers, max_seq_length)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, vocab_size


def load_tokenizer(tokenizer_path):
    return Tokenizer.from_file(tokenizer_path)

def predict_next_tokens(model, tokenizer, prompt, num_tokens, temperature=1.0):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    for _ in range(num_tokens):
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            outputs = model(input_ids)
        scaled_logits = outputs[0, -1, :] / temperature
        probabilities = F.softmax(scaled_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, 1).item()
        tokens.append(next_token_id)
    return tokenizer.decode(tokens)


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Token prediction with a Transformer model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer file.")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt text file.")
    parser.add_argument("--num_tokens", type=int, required=True, help="Number of tokens to predict.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for prediction.")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length.")
    args = parser.parse_args()

    model, vocab_size = load_model(args.model_path, args.embed_dim, args.num_heads, args.num_layers, args.max_seq_length)
    tokenizer = load_tokenizer(args.tokenizer_path)

    with open(args.prompt_path, 'r') as file:
        prompt = file.read().strip()

    predicted_text = predict_next_tokens(model, tokenizer, prompt, args.num_tokens, args.temperature)
    print("Predicted text:", predicted_text)

if __name__ == "__main__":
    main()
