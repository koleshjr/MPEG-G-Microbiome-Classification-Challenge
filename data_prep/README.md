# Data Preparation

This repository provides a robust pipeline for preparing microbiome sequencing data for machine learning and modeling. The workflow supports both `.mgb` and `.fastq` files, and extracts subtle, high-value features from raw reads using efficient multiprocessing.

---

## Directory Structure

Organize your data as follows:

```
Data/
├── TrainFiles/
│     └── sample.mgb
├── TestFiles/
│     └── sample.mgb
├── TrainFastQ/
│     └── sample.fastq
├── TestFastQ/
│     └── sample.fastq
|── Train.csv
|── Test.csv
```

- If you have `.mgb` files, store them in `TrainFiles/` and `TestFiles/`.
- If you have `.fastq` files, store them in `TrainFastQ/` and `TestFastQ/`.

---

## Advanced Feature Extraction

The script [`prepare_data_multiprocessing.py`](prepare_data_multiprocessing.py) processes FASTQ files to extract a comprehensive set of features for each sample, enabling subtle and powerful modeling. Key aspects include:

- **Multiprocessing:** Efficiently processes large datasets using all available CPU cores.
- **Resumable Processing:** Tracks processed files with log files, allowing safe interruption and restart.
- **Feature Types:**
  - **Basic statistics:** Number of reads, average read length, GC content, nucleotide fractions (A, T, G, C).
  - **Quality metrics:** Fractions of bases with Q20 and Q30 phred scores.
  - **K-mer features:** For k=2,3,4,5, computes normalized frequencies for the most common k-mers (canonical only, configurable).
  - **Customizable vocabularies:** Builds k-mer vocabularies from training data, ensuring consistent feature sets.

---

## Usage

1. **Prepare your data:** Place FASTQ files in the appropriate directories as shown above.
2. **Run the script:**
   ```bash
   python data_prep/prepare_data_multiprocessing.py
   ```
3. **Outputs:**
   - Processed feature CSVs:  
     - `Data/ProcessedFiles/train_features_with_kmers_new.csv`
     - `Data/ProcessedFiles/test_features_with_kmers_new.csv`
   - K-mer vocabulary:  
     - `Data/kmer_vocab_new.json`
   - Logs of processed files:  
     - `Data/train_processed_new.log`
     - `Data/test_processed_new.log`

---

## Customization

- **K-mer settings:**  
  Edit `K_DICT` in the script to change k-mer sizes and vocabulary sizes.
- **Input/output paths:**  
  Adjust the config section at the top of `prepare_data_multiprocessing.py` to match your directory structure.
- **Canonical k-mers:**  
  Set `CANONICAL_ONLY` to `False` to include non-canonical k-mers.

---

## Notes

- The pipeline is designed for robustness and scalability, handling interruptions and large datasets gracefully.
- Feature extraction is tailored for microbiome classification tasks, but can be adapted for other genomics applications.
- CSV outputs are suitable for direct use in machine learning frameworks.

---

## Example Feature Columns

- `file`, `num_reads`, `avg_read_len`, `gc_content`, `q20_fraction`, `q30_fraction`, `A`, `T`, `G`, `C`
- `k2_AA`, `k2_AC`, ..., `k5_ACGTG`, ... (hundreds of k-mer frequency columns)

---

## Requirements

### Install dependencies with `uv`

First, install `uv` if you don't have it:
```bash
pip install uv
```

create an environment:
```bash
uv venv
source .venv/bin/activate
```

Install dependencies:
```bash
uv sync
```