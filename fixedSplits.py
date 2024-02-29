# fixedSplits.py

import argparse

def fixed_length_split(genome, chunk_size):
    """Split a genome into fixed-length chunks."""
    return [genome[i:i + chunk_size] for i in range(0, len(genome), chunk_size)]

def process_genomes(input_file, output_file, chunk_size):
    with open(input_file, 'r') as file:
        genomes = file.readlines()

    chunks = []
    for genome in genomes:
        gene_names = genome.strip().split()
        chunks.extend(fixed_length_split(gene_names, chunk_size))

    # Writing to output file
    with open(output_file, 'w') as file:
        for chunk in chunks:
            file.write(' '.join(chunk) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Split genome datasets into fixed-size chunks.")
    parser.add_argument("input_file", help="Path to the input genome file")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("chunk_size", type=int, help="Chunk size for fixed-length splits")
    args = parser.parse_args()

    process_genomes(args.input_file, args.output_file, args.chunk_size)

if __name__ == "__main__":
    main()
