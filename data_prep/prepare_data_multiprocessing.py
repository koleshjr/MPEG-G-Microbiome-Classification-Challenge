import os
import json
import csv
import math
from pathlib import Path
from collections import Counter
from typing import Dict, List

from Bio import SeqIO
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
import traceback

# ----------------------------
# CONFIG
# ----------------------------
VOCAB_JSON = "kmer_vocab_new.json"             # saved/loaded vocab
PROCESSED_LOG = "processed_files.log"      # list of processed filenames (one per line)

K_DICT = {2: 16, 3: 64, 4: 256, 5: 256}   # all 4-mers + top 256 5-mers
MAX_READS_PER_FILE_FOR_VOCAB = 200_000
MAX_READS_PER_FILE_FOR_FEATURES = None
CANONICAL_ONLY = True
NUM_WORKERS =os.cpu_count() or 4   # parallel workers

# ----------------------------
# UTILITIES
# ----------------------------
def is_valid_kmer(kmer: str) -> bool:
    return all(ch in "ACGT" for ch in kmer) if CANONICAL_ONLY else True

def list_fastq_files(fastq_dir: str) -> List[str]:
    return sorted([fn for fn in os.listdir(fastq_dir) if fn.endswith((".fastq", ".fq"))])

def load_processed_set(log_path: str) -> set:
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_processed(log_path: str, filename: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

# ----------------------------
# VOCAB BUILDING
# ----------------------------
def build_kmer_vocab(fastq_dir: str, k_dict: Dict[int, int], max_reads_per_file=None) -> Dict[int, List[str]]:
    kmer_counts = {k: Counter() for k in k_dict}
    files = list_fastq_files(fastq_dir)

    for fn in tqdm(files, desc="Building k-mer vocabulary"):
        path = os.path.join(fastq_dir, fn)
        for i, record in enumerate(SeqIO.parse(path, "fastq")):
            seq = str(record.seq).upper()
            L = len(seq)
            for k in k_dict:
                if L < k: continue
                for j in range(L - k + 1):
                    kmer = seq[j:j+k]
                    if is_valid_kmer(kmer):
                        kmer_counts[k][kmer] += 1
            if max_reads_per_file and (i + 1) >= max_reads_per_file:
                break

    vocab = {}
    for k, top_n in k_dict.items():
        most_common = kmer_counts[k].most_common(top_n)
        vocab[k] = [kmer for kmer, _ in most_common]
    return vocab

def save_vocab(vocab: Dict[int, List[str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in vocab.items()}, f)

def load_vocab(path: str) -> Dict[int, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}

# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
def extract_features_with_vocab(path: str, vocab: Dict[int, List[str]], max_reads_per_file=None) -> dict:
    read_count, total_len, gc_total = 0, 0, 0
    nt_counter = Counter()
    q20_count = q30_count = total_bases = 0
    kmer_counters = {k: Counter() for k in vocab}

    for i, record in enumerate(SeqIO.parse(path, "fastq")):
        seq = str(record.seq).upper()
        L = len(seq)
        if L == 0: continue
        read_count += 1
        total_len += L
        gc_total += seq.count("G") + seq.count("C")
        nt_counter.update(ch for ch in seq if ch in "ATGC")

        quals = record.letter_annotations.get("phred_quality", [])
        total_bases += len(quals)
        q20_count += sum(q >= 20 for q in quals)
        q30_count += sum(q >= 30 for q in quals)

        for k in vocab:
            if L < k: continue
            for j in range(L - k + 1):
                kmer = seq[j:j+k]
                if is_valid_kmer(kmer) and kmer in vocab[k]:
                    kmer_counters[k][kmer] += 1

        if max_reads_per_file and (i + 1) >= max_reads_per_file:
            break

    if read_count == 0 or total_len == 0:
        return None

    features = {
        "num_reads": read_count,
        "avg_read_len": total_len / read_count,
        "gc_content": gc_total / total_len,
        "q20_fraction": q20_count / total_bases if total_bases else 0,
        "q30_fraction": q30_count / total_bases if total_bases else 0,
        "A": nt_counter["A"] / total_len,
        "T": nt_counter["T"] / total_len,
        "G": nt_counter["G"] / total_len,
        "C": nt_counter["C"] / total_len,
    }

    for k, kmers in vocab.items():
        total_kmers_k = sum(kmer_counters[k].values())
        for kmer in kmers:
            features[f"k{k}_{kmer}"] = (
                kmer_counters[k][kmer] / total_kmers_k if total_kmers_k else 0.0
            )
    return features

# ----------------------------
# PROGRESSIVE CSV WRITER
# ----------------------------
def ensure_csv_header(csv_path: str, header: List[str]) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def append_row(csv_path: str, row_dict: dict, header: List[str]) -> None:
    row = [row_dict.get(col, "") for col in header]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# ----------------------------
# MULTIPROCESSING HELPERS
# ----------------------------
def _worker_task(args):
    task_type, fn, fastq_dir, *rest = args
    try:
        path = os.path.join(fastq_dir, fn)
        if task_type == "kmer":
            vocab, header = rest
            feats = extract_features_with_vocab(path, vocab, max_reads_per_file=MAX_READS_PER_FILE_FOR_FEATURES)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        if feats is None:
            return None, fn
        feats["file"] = fn
        return feats, fn

    except Exception as e:
        print(f"❌ Error in worker for {fn}: {e}")
        traceback.print_exc()
        return None, fn


def process_dataset(fastq_dir, output_csv, processed_log, vocab, num_workers=NUM_WORKERS):
    base_cols = ["file","num_reads","avg_read_len","gc_content",
                 "q20_fraction","q30_fraction","A","T","G","C"]
    kmer_cols = [f"k{k}_{kmer}" for k in sorted(vocab) for kmer in vocab[k]]
    header = base_cols + kmer_cols
    ensure_csv_header(output_csv, header)

    processed = load_processed_set(processed_log)
    files = list_fastq_files(fastq_dir)
    todo = [fn for fn in files if fn not in processed]
    print(f"🔍 {len(todo)} files to process in {fastq_dir}")

    if not todo: return

    try:
        with mp.Pool(processes=num_workers) as pool:
            for feats, fn in tqdm(pool.imap_unordered(
                _worker_task, [('kmer', fn, fastq_dir, vocab, header) for fn in todo]),
                total=len(todo), desc=f"Extracting {fastq_dir}"):
                if feats:
                    append_row(output_csv, feats, header)
                append_processed(processed_log, fn)
    except Exception as e:
        print(f"⚠️ Multiprocessing failed ({e}), falling back to sequential.")
        for fn in tqdm(todo, desc=f"Sequential {fastq_dir}"):
            feats, _ = _worker_task((fn, fastq_dir, vocab, header))
            if feats:
                append_row(output_csv, feats, header)
            append_processed(processed_log, fn)

    print(f"✅ Finished {fastq_dir} → {output_csv}")

# ----------------------------
# MAIN PIPELINE
# ----------------------------
TRAIN_DIR = "Data/TrainFastQ"
TEST_DIR = "DataTestFastQ"
TRAIN_OUTPUT_CSV = "Data/ProcessedFiles/train_features_with_kmers_new.csv"
TEST_OUTPUT_CSV = "Data/ProcessedFiles/test_features_with_kmers_new.csv"
VOCAB_JSON = "Data/kmer_vocab_new.json"
TRAIN_LOG = "Data/train_processed_new.log"
TEST_LOG = "Data/test_processed_new.log"

def main():
    if os.path.exists(VOCAB_JSON):
        vocab = load_vocab(VOCAB_JSON)
        print(f"🔍 Loaded vocab from {VOCAB_JSON}")
    else:
        vocab = build_kmer_vocab(TRAIN_DIR, k_dict=K_DICT,
                                 max_reads_per_file=MAX_READS_PER_FILE_FOR_VOCAB)
        save_vocab(vocab, VOCAB_JSON)

    process_dataset(TEST_DIR, TEST_OUTPUT_CSV, TEST_LOG, vocab, num_workers=NUM_WORKERS)
    process_dataset(TRAIN_DIR, TRAIN_OUTPUT_CSV, TRAIN_LOG, vocab, num_workers=NUM_WORKERS)


if __name__ == "__main__":
    main()
