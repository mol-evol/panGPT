import argparse

def moving_window_split(genome, window_size, shift_size):
    """Split a genome using a moving window."""
    return [genome[i:i + window_size] for i in range(0, len(genome) - window_size + 1, shift_size)]

def process_genomes(input_file, output_file, window_size, shift_size):
    with open(input_file, 'r') as file:
        genomes = file.readlines()

    chunks = []
    for genome in genomes:
        gene_names = genome.strip().split()
        chunks.extend(moving_window_split(gene_names, window_size, shift_size))

    # Writing to output file
    with open(output_file, 'w') as file:
        for chunk in chunks:
            file.write(' '.join(chunk) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Split genome datasets using a moving window.")
    parser.add_argument("input_file", help="Path to the input genome file")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("window_size", type=int, help="Window size for moving-window splits")
    parser.add_argument("shift_size", type=int, help="Shift size for moving-window splits")
    args = parser.parse_args()

    process_genomes(args.input_file, args.output_file, args.window_size, args.shift_size)

if __name__ == "__main__":
    main()
