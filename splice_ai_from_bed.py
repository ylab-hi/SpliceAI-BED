import pysam
from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np
import argparse

# command line arguments for bed file input, genome location, and output file, using argparse
parser = argparse.ArgumentParser(description='Predict splice sites from BED file regions')
parser.add_argument('--bed', required=True, help='Input BED file')
parser.add_argument('--genome', required=True, help='Reference genome FASTA file')
parser.add_argument('--output', required=True, help='Output file path')
args = parser.parse_args()


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement[base] for base in reversed(seq))


def predict_single_region(chrom, start, end, strand, models):
    """
    Predict splice sites for a single genomic region using chunks
    
    Parameters:
    chrom (str): Chromosome name
    start (int): Start position
    end (int): End position  
    strand (str): '+' or '-' for strand
    models (list): List of loaded SpliceAI models
    
    Returns:
    numpy array: Array of shape (region_length, 3) containing null/acceptor/donor probabilities
    """
    # add padding to start and end
    start = start - 10
    end = end + 10
    
    # 5kb chunks
    chunk_size = 5000
    region_length = end - start
    context = 10000
    
    # Initialize array for final predictions 
    final_preds = np.zeros((region_length, 3))
    
    # Calculate number of chunks needed
    num_chunks = (region_length - 1) // chunk_size + 1
    
    print(f"Processing {num_chunks} chunks for region {chrom}:{start}-{end}")
 
    for i in range(num_chunks):
        print(f"Processing chunk {i+1}/{num_chunks}")
        # Get coordinates for this chunk
        chunk_start = start + (i * chunk_size)
        chunk_end = min(chunk_start + chunk_size, end)
        
        # Get sequence with padding
        seq_start = max(0, chunk_start - (context//2))
        seq_end = min(genome.get_reference_length(chrom), chunk_end + (context//2))
        
        # Add padding if needed
        left_pad = max(0, context//2 - (chunk_start - seq_start))
        right_pad = max(0, context//2 - (seq_end - chunk_end))
        
        seq = ('N' * left_pad +
               genome.fetch(chrom, seq_start, seq_end).upper() +
               'N' * right_pad)
        
        if strand == '-':
            seq = reverse_complement(seq)
            
        # One-hot encode
        X = one_hot_encode(seq)[np.newaxis, :]
        
        # Get predictions from all models
        chunk_preds = []
        for model in models:
            pred = model.predict(X, verbose=0)
            if strand == '-':
                pred = np.flip(pred, axis=1)
            chunk_preds.append(pred)

        # Average predictions across models
        avg_pred = np.mean(chunk_preds, axis=0)  # Shape should be (1, chunk_length, 3)
        
        # Extract relevant portion (removing padding) 
        chunk_length = chunk_end - chunk_start
        relevant_preds = avg_pred[0]  # Remove batch dimension
        
        # Add to final predictions array
        start_idx = i * chunk_size
        final_preds[start_idx:start_idx+chunk_length] = relevant_preds
    
    if strand == '+':
        acceptor_probs = final_preds[:, 1][10:-10]  # Column 1 for acceptor probabilities
        donor_probs = final_preds[:, 2][8:-12]     # Column 2 for donor probabilities
    else:
        acceptor_probs = final_preds[:, 1][8:-12]
        donor_probs = final_preds[:, 2][10:-10]
        
    return donor_probs, acceptor_probs

def load_bed_file(bed_file):
    """
    Load a BED file and yield regions
    
    Parameters:
    bed_file (str): Path to BED file
    
    Yields:
    Tuples (chrom, start, end, strand)
    
    Raises:
    ValueError: If strand information is missing
    """
    with open(bed_file) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 4:
                raise ValueError(f"BED file should be formatted as: chrom start end strand")
            chrom, start, end, strand = fields[:4]
            if strand not in ('+', '-'):
                raise ValueError(f"Strand should be either '+' or '-'")
            start, end = int(start), int(end)
            yield (chrom, start, end, strand)

# load genome
genome = pysam.FastaFile(args.genome)

# Load models once
paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x)) for x in paths]

bed_file = args.bed
output_file = args.output

with open(output_file, 'w') as out:
    for chrom, start, end, strand in load_bed_file(bed_file):
        donor_probs, acceptor_probs = predict_single_region(chrom, start, end, strand, models)
        for i, (donor_prob, acceptor_prob) in enumerate(zip(donor_probs, acceptor_probs), start=start):
            out.write(f"{chrom}\t{i}\t{i+1}\t{donor_prob}\t{acceptor_prob}\n")

"""

preds = predict_single_region("chr17", 7727293,7727293+1, "+", models)
acceptor_probs = preds[:, 1]  # Column 1 for acceptor probabilities 
donor_probs = preds[:, 2]     # Column 2 for donor probabilities

127,735,069-127,743,509
40,151,609-40,160,804
201,159,632-201,179,868

chr17:7,687,148-7,687,709
chr17:7,732,841-7,733,504
7,723,491-7,723,906

pos = 7734294
preds = predict_single_region("chr17", pos,pos + 1, "+", models)

# subtract position from end of interval
# neg strand: acceptor +1 
# neg strand: donor - 1

# add position from start of interval
# pos strand: acceptor + 0
# pos strand: donor + 2


# test cases
# + strand
preds = predict_single_region("chr17", 7727293,7727293+1, "+", models)
preds = predict_single_region("chr17", 7727121,7727121+1, "+", models)
preds = predict_single_region("chr7", 55189283,55196151+1, "+", models)

# - strand
preds = predict_single_region("chr17", 7670716,7670716+1, "-", models)
preds = predict_single_region("chr17", 7670608,7670608+1, "-", models)

preds = predict_single_region("chr1", 235799729,235801790+1, "-", models) 
preds = predict_single_region("chr1", 235796663,235802632+1, "-", models)
chr1:235,796,663-235,802,632

"""
