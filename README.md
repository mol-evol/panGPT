
# README File for panGPT v0.10a and panPrompt v0.01.


Developed by: James McInerney

Website: [pangenome.ai](http://pangenome.ai)

Email: panGPTprogram@gmail.com

License: GNU General Public License v4.0

NOTE: This is BETA software and may have bugs.

### panGPT and panPrompt: Generative Pre-Trained Transformer for Large Pangenome Models (LPMs) from scratch.

---
#### Introduction
<b>panGPT</b> is a Python program designed to train a transformer model on pangenomic presence-absence data.

<b>panPrompt</b> is a python program that uses the model from panGPT to predict the next genes on the basis of a prompt.

This <b>README</b> file aims to explain how to use the programs and understand their various command-line options.

#### Requirements
- Python 3.x
- PyTorch
- Tokenizers library
- math library

## Installation Guide for panGPT and panPrompt <u>with</u> Conda
This guide provides step-by-step instructions on how to install `panGPT`, `panPrompt` and their relevant libraries using Conda.

#### Prerequisites
Ensure that you have Conda installed on your system. If Conda is not installed, you can download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).

#### Using Conda

#### Step 1: Create a Conda Environment
It's recommended to create a new Conda environment for `panGPT` and `panPrompt` to avoid conflicts with existing libraries. Here I call this environment `panAI`, but you can call it whatever makes sense for you.

```bash
conda create -n panAI python=3.8
```

#### Step 2: Activate the virtual environment

```bash
conda activate panAI
```
#### Step 3: Install PyTorch and the Tokenizers library
Install PyTorch in the Conda environment. Make sure to choose the version compatible with your system's CUDA version if you plan to use GPU acceleration (please take advice from your local sysadmin on this).

```bash
conda install pytorch cudatoolkit=10.2 -c pytorch
```

Install the tokenizers library, which is required for text tokenization.
```bash
conda install tokenizers
```

#### Step 4: Clone the repository:

   ```bash
   git clone https://github.com/mol-evol/panGPT.git
```


## Installation Guide for panGPT <u>without</u> Conda

This guide provides instructions on how to install `panGPT` and `panPrompt` and their required libraries using `pip`, Python's package manager.

#### Prerequisites

Ensure that Python (preferably version 3.6 or newer) is installed on your system. You can download and install Python from [python.org](https://www.python.org/downloads/).

#### Step 1: Create a Virtual Environment

It's recommended to create a virtual environment for `panGPT` to avoid conflicts with existing Python libraries.

```bash
python -m venv panAI -env
```
#### Step 2: Activate the virtual environment
```bash
source pangpt-env/bin/activate
```
#### Step 3: Install PyTorch and the Tokenizers library
Install PyTorch using pip. Visit the PyTorch Get Started page to find the installation command that matches your system configuration (e.g., with or without GPU support).
```bash
pip install torch tokenizers
```

#### Step 4: Clone the repository:

   ```bash
   git clone https://github.com/mol-evol/panGPT.git
```

or alternatively download the panGPT program from

https://github.com/mol-evol/


## Generating a Large Pangenome Model (LPM) from scratch using panGPT.


panGPT implements a simple Transformer model with positional encoding that takes as input a pangenome file with each genome on a single line. Each gene name is separated by a space and gene names are consistent across all genomes - i.e. homologs or orthologs all get the same name.

The LPM is then trained on these data, using the Transformers approach outlined in Vaswani et al., (2017).

#### Input file format
The input file format for the pangenome dataset is simply a set of gene names, with a single genome being on a single line.  The file can be easily generated from the output of programs such as [Roary](https://sanger-pathogens.github.io/Roary/).

e.g.:

        `
        atpH group_44044 group_43943 group_43935 frdA [...]
        group_12592 FtsZ frdA group_87657 atpH [...]
        [...]
        `

The first line contains genome 1, the second line contains genome 2, etc.


The accompanying file [`inputPangenomes.txt`](https://github.com/mol-evol/panGPT/blob/main/inputPangenomes.txt) is from Beavan and McInerney ([2024](https://www.pnas.org/doi/abs/10.1073/pnas.2304934120)).


## Command-Line Options

1. `--input_file`:
   - Type: String
   - Required: Yes
   - Description: Path to the input file containing pangenome data. This file should have pangenomic data formatted with each line representing a genome, with each gene represented by its family name, and the order of the family names is the order in which they occur in the individual genomes. Because this particular program uses positional information, the order of the genes needs to be preserved.

2. `--model_type`:
    - Type: String
    - Default: "transformer"
    - Description: Type of model architecture (transformer or reformer)

3. `--longformer_attention_window`:
    - Type: Integer
    - Default: 512
    - Description: Attention window size in the Longformer model. Not relevant for Transformer model

4. `--embed_dim`:
   - Type: Integer
   - Default: 256
   - Description: The dimension of the embedding layer in the transformer model. Think of this as setting the size of the feature set that the model uses to understand the data. I think of it as the number of "learned features" that might explain the context for the presence or absence of the focal gene. Genes with similar requirements for their presence or absence in a genome would have similar vector representations. Larger embedding sizes can capture more complex patterns but at the cost of higher computational requirements and the risk of overfitting. Smaller sizes are computationally efficient but might not capture the nuances in the data. These embedding dimensions are learned during training and are not manually entered features.

5. `--num_heads`:
   - Type: Integer
   - Default: 8
   - Description: The number of attention heads in the transformer model. Each head helps the model to focus on different parts of the input data. For example, if you set an embedding size of `512` and num_heads is set to `8`, each head would operate on a 64-dimensional space (`512 / 8`). Each head independently attends to the input sequence, and their outputs are then concatenated and linearly transformed into the expected output dimension.

6. `--num_layers`:
   - Type: Integer
   - Default: 4
   - Description: The number of layers that make a stack in the transformer model. More layers can potentially lead to a deeper understanding of the data but may require more computational resources. Each layer in a Transformer typically consists of a multi-head self-attention mechanism and a position-wise feed-forward network. When you stack multiple such layers on top of each other the output of one layer becomes the input to the next layer.

7. `--max_seq_length`:
   - Type: Integer
   - Default: 256
   - Description: The maximum length of sequences that the model will process. Sequences longer than this will be truncated.

8. `--batch_size`:
   - Type: Integer
   - Default: 32
   - Description: The number of sequences processed at one time during training and validation. Larger batch sizes can speed up training but require more memory, smaller batch sizes introduce more stochasticity and might help prevent overfitting, at the cost of slower training times.

9. `--model_dropout_rate`:
    - Type: Float
    - Default: 0.1
    - Description: The dropout rate is the probability of a neuron being dropped out during training. Dropout is a regularization technique where randomly selected neurons are ignored during training. They are "dropped-out" randomly. Dropout helps prevent overfitting by introducing noise in the network, forcing nodes in the network to probabilistically take on more or less responsibility for the inputs. A value of 0.1 means there is a 10% chance of a neuron being dropped out during each training iteration.

10. `--learning_rate`:
   - Type: Float
   - Default: 0.0001
   - Description: The learning rate for the optimizer. This sets how much the model adjusts its weights in response to the estimated error each time the model weights are updated.

11. `--lr_scheduler_factor`:
    - Type: Float
    - Defauly: 0.5
    - Description: Factor by which the learning rate will be reduced by the learning rate scheduler (i.e. if triggered, the learning rate is multiplied by this factor).

12. `--lr_patience`:
   - Type: Integer
   - Default: 10
   - Description: The number of epochs with no improvement on validation loss after which the learning rate will be reduced by the `--lr_scheduler_factor` .

13. `--weight_decay`:
   - Type: Float
   - Default: 1e-4
   - Description: The weight decay parameter for the optimizer. It's a regularization technique to prevent overfitting by penalizing large weights. It encourages the program to learn simpler patterns, thereby improving its ability to generalize to new data.

14. `--early_stop_patience`:
    - Type: Integer
    - Default: 20
    - Description: The number of epochs with no improvement on validation loss after which the training will stop. This is part of the early stopping mechanism to prevent overfitting.

15. `--min_delta`:
    - Type: Float
    - Default: 0.01
    - Description: The minimum change in validation loss to qualify as an improvement. This is used in conjunction with the `patience` parameter for early stopping. Lower delta values will cause the program to train for longer, while higher values will cause the program to stop earlier.

16. `--epochs`:
    - Type: Integer
    - Default: 50
    - Description: The maximum number of complete passes through the training dataset. The program will run for a maximum number of epochs, but if learning is not improving, then the early-stopping mechanism will stop the program before this limit is reached.

17. `--max_vocab_size`:
    - Type: Integer
    - Default: 70000
    - Description: The maximum size of the vocabulary. Tokens beyond this size will be mapped to an 'unknown' token. If you wish to use all "words" in your pangenome, then set this value to be larger than the total number of gene families. If you wish to cut off a certain number of low-frequency gene families, then set this vocab to a lower value and it will only use the most frequent gene families up to this limit.

18. `--model_save_path`:
    - Type: String
    - Default: "model_checkpoint.pth"
    - Description: The path where the model checkpoints will be saved.

19. `--tokenizer_file`:
    - Type: String
    - Default: "pangenome_gpt_tokenizer.json"
    - Description: The filename for saving and loading the tokenizer.

20. `--train_size`:
    - Type: Float
    - Default: 0.8
    - Description: The proportion of the dataset to include in the training set. This value should be between 0 and 1. The remaining data will be split between validation and test sets.

21. `--val_size`:
    - Type: Float
    - Default: 0.1
    - Description: The proportion of the dataset to include in the validation set. This value should be between 0 and 1. The remaining data, after subtracting the training and validation sets, will be used as the test set.

22. `--seed`:
    - Type: Integer
    - Default: 42
    - Description: A number that can be used to seed a random number generator.

23. `--pe_max_len`:
    - Type: Integer
    - Default: 5000
    - Description: The maximum length to be used for positional encoding. This should be at least as long as the max_seq_length command line argument.

24. `--pe_dropout_rate`:
    - Type: Float
    - Default: 0.1
    - Description: The Dropout Rate for positional encoding. This is a regularization technique, preventing the model from depending too much on specific positional information. A proportion of the inputs are set to zero during trainign and this introduces noise that helps prevent the model form overfitting.

25. `--log_dir`:
    - Type: String
    - Default: logs
    - Description: The directory name where the log files are stored for Tensorboard.


---

#### Usage Notes
- To run the program, you need to provide the `--input_file` with the path to your pangenome data file as a minimum.
- Adjust the other parameters based on your dataset's characteristics and the computational resources available to you.
- Monitor the output for messages about training progress and any possible issues.

---

## Token Prediction with Simple Transformer Model program panPrompt

### Overview
This Python program carries out next token prediction using the Transformer model from panGPT. panPrompt takes a prompt file as input and then produces the next <b><i>`n`</i></b> tokens, as specified by the user.

### Features
- **Positional Encoding**: Implements positional encoding to inject sequence order information into the model.
- **Transformer Model**: Uses a basic Transformer model architecture for sequence generation tasks.
- **Token Prediction**: Functionality for predicting a sequence of tokens given an input prompt.


#### Usage
 The program can be executed via the command line, allowing users to input a model checkpoint, tokenizer file, prompt file, and other parameters for token prediction.

Execute the program from the command line using a prompt with this general format:

   ```bash
    python panPrompt.py --model_path "path/to/model_checkpoint.pth" \
                   --tokenizer_path "path/to/tokenizer.json" \
                   --prompt_path "path/to/prompt.txt" \
                   --num_tokens 50 \
                   --temperature 1.0 \
                   --embed_dim 256 \
                   --num_heads 8 \
                   --num_layers 4 \
                   --max_seq_length 256
```

#### Command Line Options

The script accepts the following command line arguments:
```bash
- `--model_path`: Path to the trained model checkpoint.
     - e.g. Usage: `--model_path "path/to/model_checkpoint.pth"`

- `--tokenizer_path`: Path to the tokenizer file.
     -e.g.  Usage: `--tokenizer_path "path/to/tokenizer.json"`

- `--prompt_path`: Path to the text file containing the input prompt.
     -e.g.  Usage: `--prompt_path "path/to/prompt.txt"`

- `--num_tokens`: Number of tokens to predict.
     -e.g.  Usage: `--num_tokens 50`

- `--temperature`: Controls the randomness of predictions. Default is 1.0.
     -e.g.  Usage: `--temperature 1.0`

- `--embed_dim`: Embedding dimension size. Default is 256.
     -e.g.  Usage: `--embed_dim 256`

- `--num_heads`: Number of attention heads. Default is 8.
     -e.g.  Usage: `--num_heads 8`

- `--num_layers`: Number of layers in the transformer. Default is 4.
     -e.g.  Usage: `--num_layers 4`

- `--max_seq_length`: Maximum length of the input sequences. Default is 256.
     -e.g.  Usage: `--max_seq_length 256`
```

For example to run the program using a given model, tokenizer and prompt you might use this command:

```bash
python panPrompt.py --model_path "models/model.pth" \
          --tokenizer_path "tokenizers/tokenizer_gpt_pangenome.json" \
          --prompt_path "prompts/prompt.txt" \
          --num_tokens 50
```

In this case, you would have your model stored (in pytorch format) in a subdirectory called models, the pangenome-specific tokenizer stored in a subdirectory called tokenizers, and your prompt file is stored in a subdirectory called prompts. panPrompt.py takes these three files and uses them to make predictions about the next tokens - in this case you have asked for 50 tokens (protein-coding gene families in this case).

---
### Notes


#### Training model format

The trained model is output in pytorch format. The checkpoint files store

           "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,

This means that you can restart the training run from the checkpoint file, in those cases where either some error caused the program to stop running, or if you want to extend the run (e.g. run for more epochs, or change parameters for fine tuning).


---

Disclaimer
This code is provided 'AS IS' without any warranties of any kind, including, but not limited to, its fitness for a particular purpose. The author disclaims all liability for any damages, direct or indirect, resulting from the use of this code.

---

For more information, visit [pangenome.ai](http://pangenome.ai/).


#### References:

Beavan, A. J. S., et al. (2024). "Contingency, repeatability, and predictability in the evolution of a prokaryotic pangenome." ***Proceedings of the National Academy of Sciences USA*** 121(1): e2304934120.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I., 2017. "Attention is all you need". ***Advances in neural information processing systems***, 30.

Beltagy, I., Peters, M.E. and Cohan, A., 2020. Longformer: The long-document transformer. ***arXiv preprint*** arXiv:2004.05150.
