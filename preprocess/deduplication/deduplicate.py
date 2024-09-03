import zstandard as zstd
import io
import jsonlines
import json
import os
import shutil
from typing import List, Dict
from multiprocessing import Pool
from preprocess.deduplication.lshfilter import DedupLSHFile, DedupLSH
import logging
from multiprocessing import current_process
import signal
N_THREADS = 24
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
# Define input and output directories
input_dir = "D:/Elham/EnrichMyData/EnrichData_PC/data/labeled/"
output_dir = "D:/Elham/EnrichMyData/EnrichData_PC/data/labeled/processed/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


def read_jsonl_z(file_path):
    """Read and decompress .jsonl.z file, return list of dictionaries."""
    data = []
    with open(file_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.stream_reader(compressed_file)
        
        with io.TextIOWrapper(decompressed, encoding='utf-8') as reader:
            for obj in jsonlines.Reader(reader):
                data.append(obj)
    return data

def decompress_input_file(input_file, intermediate_file):
    """Decompress a single file and save to intermediate .jsonl."""
    try:
        data = read_jsonl_z(input_file)
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            for obj in data:
                json.dump(obj, f)
                f.write('\n')
        print(f"Decompressed file written to {intermediate_file}")
    except Exception as e:
        print(f"Error decompressing file {input_file}: {e}")
        raise

def recompress_output_file(intermediate_file, output_compressed):
    """Recompress the deduplicated file."""
    try:
        with open(intermediate_file + ".dedup", 'rb') as f_in:
            dctx = zstd.ZstdCompressor()
            with open(output_compressed, 'wb') as f_out:
                shutil.copyfileobj(dctx.stream_reader(f_in), f_out)
        print(f"Recompressed file written to {output_compressed}")
    except Exception as e:
        print(f"Error recompressing file {intermediate_file}: {e}")
        raise

def perform_deduplication(input_file, intermediate_file, deduper):
    """Perform deduplication on a single file."""
    try:
        # Initialize DedupLSHFile
        dedup_file = DedupLSHFile(
            input_file=intermediate_file,
            output_file=intermediate_file + ".dedup",
            skip_header=False,  # Adjust based on your file
            batch_size=1024
        )

        # Perform deduplication
        with Pool(N_THREADS, initializer=init_worker) as pool:
            dedup_file.dedup(deduper, pool)
        print(f"Deduplication completed for {input_file}.")
    except Exception as e:
        print(f"Error during deduplication for {input_file}: {e}")
        raise

def process_files(input_dir, output_dir, deduper):
    """Process all files in the input directory."""
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.jsonl.z'):
            input_file = os.path.join(input_dir, file_name)
            intermediate_file = os.path.join(input_dir, file_name[:-6])  # Remove .z extension
            output_compressed = os.path.join(output_dir, file_name)
            
            # Decompress the input file
            decompress_input_file(input_file, intermediate_file)
            
            # Perform deduplication
            perform_deduplication(input_file, intermediate_file, deduper)
            
            # Recompress the output file
            recompress_output_file(intermediate_file, output_compressed)
            
            # Clean up intermediate files
            try:
                os.remove(intermediate_file)
                os.remove(intermediate_file + ".dedup")
            except Exception as e:
                print(f"Error cleaning up intermediate files for {input_file}: {e}")

# Initialize DedupLSH
deduper = DedupLSH(
    threshold=0.9,
    n_perm=128,
    ngram_range=(1, 3),
    min_char_len=10,
    token_len=128,
    preprocess_text=True,
    reset_after=10000  # example value
)

if __name__ == "__main__": 
    process_files(input_dir, output_dir, deduper)