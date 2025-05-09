#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read a CSV file, call the MedQA model to generate answers,and write the results into the CSV.
"""

import argparse
import pandas as pd
from MedQA_fine_tune import generate_answer  

def main():
    parser = argparse.ArgumentParser(description="Generate RAG responses for the 'Question' column in a test CSV")
    parser.add_argument(
        "--input_csv", "-i", 
        type=str, required=True, 
        help="Path to the input CSV file; must contain the 'Question' column"
    )
    parser.add_argument(
        "--output_csv", "-o", 
        type=str, default="output_with_rag.csv", 
        help="Path to the output CSV file; a new 'RAG Response' column will be added"
    )
    parser.add_argument(
        "--max_input_len", type=int, default=128,
        help="Maximum input length for generation"
    )
    parser.add_argument(
        "--max_output_len", type=int, default=512,
        help="Maximum output length for generation"
    )
    parser.add_argument(
        "--beam_size", type=int, default=5,
        help="Beam count for Beam Search"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "Question" not in df.columns:
        raise ValueError("Cannot find 'Question' column in the input CSV. Please check the file format.")

    # Initialize the result column
    df["RAG Response"] = ""

    # Generate answers
    from tqdm import tqdm
    tqdm.pandas(desc="Generating RAG responses")

    def gen_fn(question: str) -> str:
        return generate_answer(
            question=question,
            max_input_length=args.max_input_len,
            max_output_length=args.max_output_len,
            num_beams=args.beam_size
        )

    df["RAG Response"] = df["Question"].progress_apply(gen_fn)

    # Save the results
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"Results with RAG responses have been saved to `{args.output_csv}`.")

if __name__ == "__main__":
    main()
