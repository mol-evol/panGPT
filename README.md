README File for panGPT v0.02

Developed by: James McInerney
Website: [McInerney Lab](http://mcinerneylab.com/)
License: GNU General Public License v3.0

---

Introduction
panGPT is a Python program designed to train a transformer model on pangenomic presence-absence data. This README file aims to explain how to use the program and understand its various command-line options.

Command-Line Options

1. `--input_file`:
   - Type: String
   - Required: Yes
   - Description: Path to the input file containing pangenome data. This file should have genomic data formatted in a specific way that the program can read.

2. `--embed_dim`:
   - Type: Integer
   - Default: 256
   - Description: The dimension of the embedding layer in the transformer model. Think of this as setting the size of the feature set that the model uses to understand and generate the data.

3. `--num_heads`:
   - Type: Integer
   - Default: 8
   - Description: The number of attention heads in the transformer model. Each head helps the model to focus on different parts of the input data.

4. `--num_layers`:
   - Type: Integer
   - Default: 4
   - Description: The number of layers, or stacks, in the transformer model. More layers can potentially lead to a deeper understanding of the data but may require more computational resources.

5. `--max_seq_length`:
   - Type: Integer
   - Default: 256
   - Description: The maximum length of sequences that the model will process. Sequences longer than this will be truncated.

6. `--batch_size`:
   - Type: Integer
   - Default: 32
   - Description: The number of sequences processed at one time during training and validation. Larger batch sizes can speed up training but require more memory.

7. `--learning_rate`:
   - Type: Float
   - Default: 0.0001
   - Description: The learning rate for the optimizer. This sets how much the model adjusts its weights in response to the estimated error each time the model weights are updated.

8. `--weight_decay`:
   - Type: Float
   - Default: 1e-5
   - Description: The weight decay parameter for the optimizer. It's a regularization technique to prevent overfitting by penalizing large weights.

9. `--patience`:
   - Type: Integer
   - Default: 5
   - Description: The number of epochs with no improvement on validation loss after which the training will stop. This is part of the early stopping mechanism to prevent overfitting.

10. `--min_delta`:
    - Type: Float
    - Default: 0.01
    - Description: The minimum change in validation loss to qualify as an improvement. This is used in conjunction with the `patience` parameter for early stopping.

11. `--epochs`:
    - Type: Integer
    - Default: 30
    - Description: The total number of complete passes through the training dataset.

12. `--max_vocab_size`:
    - Type: Integer
    - Default: 70000
    - Description: The maximum size of the vocabulary. Tokens beyond this size will be mapped to an 'unknown' token.

13. `--model_save_path`:
    - Type: String
    - Default: "./model_checkpoint.pth"
    - Description: The path where the model checkpoints will be saved.

14. `--tokenizer_file`:
    - Type: String
    - Default: "pangenome_gpt_tokenizer.json"
    - Description: The filename for saving and loading the tokenizer.

---

Usage Notes
- To run the program, you need to provide the `--input_file` with the path to your pangenome data file.
- Adjust the other parameters based on your dataset's characteristics and the computational resources available to you.
- Monitor the output for messages about training progress and any possible issues.

---

The Obligatory Disclaimer
This code is provided 'AS IS' without any warranties of any kind, including, but not limited to, its fitness for a particular purpose. The author disclaims all liability for any damages, direct or indirect, resulting from the use of this code.

---

For more information, visit the [McInerney Lab website](http://mcinerneylab.com/).
