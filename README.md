# SpliceAI-BED
SpliceAI command-line tool only takes a VCF file as an input. In order to predict novel splice sites, it's useful to have a wrapper around SpliceAI that can predict on genomic regions more generally. 

# Installation
Install SpliceAI according to instructrions https://github.com/Illumina/SpliceAI: 

```
pip install spliceai
# or
conda install -c bioconda spliceai
```

Install `tensorflow>=1.2.0` if needed:

```
pip install tensorflow
# or
conda install tensorflow
```

SpliceAI seems to be a bit slow on CPU if you're predicting many regions. I've tested this script on A100 GPU using cuda 12.0.

## Usage 

This script takes in a BED file of genomic positions (strand information is crucial and required), and outputs a bed file where the last two columns are donor (5') and acceptor (3') probabilities respectively. Run the python script with three arguments:

```
python splice_ai_from_bed.py \
  --bed test_data/test_single_regions.bed
  --genome /path/to/hg38.fa
  --out test_data/test_single_regions_out.bed
```

If you want to predict donor/acceptor probabilites of a single position X, make sure the BED file is of the form e.g.:

```
chr1  X  X+1  +
```

This wrapper uses a single pass to predict up to 5k bases (larger regions are chunked in to 5k regions). So, for example, if you want to predict 100 sites within a 5kb region, it will be much faster to just predict the whole 5kb region than each site. 

