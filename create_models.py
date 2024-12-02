"""
Script for generating transformer models for BiomedGPT
"""

import os
import torch
from transformers import OFAModel, OFATokenizer
import json
from argparse import ArgumentParser, Namespace
import sys
import logging


with open('key_map.json', 'r') as f:
    KEY_MAP = json.load(f)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ]
)


def create_model_dir(dir:str|os.PathLike) -> None:
    """
    Create a directory to save pretrained transformer models.

    params:
    dir (os.PathLike|str): The directory to be created
    """

    try:
        os.makedirs(dir, exist_ok=False)
        logging.info(f'Model directory {dir} created successfully.')
    except OSError as e:
        logging.error(f'An error occured: {e}')


def create_model(ckpt_path:str|os.PathLike, ref_model_path:str|os.PathLike, tgt_model_path:str|os.PathLike) -> None:
    """
    Creates and saves a BiomedGPT model into the desired directory

    args:\n
    ckpt_path: The path from the model weights\n
    ref_model_path: The path to take the reference model\n
    tgt_model_path: The path to save the model
    """

    weights = torch.load(ckpt_path)

    model = OFAModel.from_pretrained(ref_model_path)
    tokenizer = OFATokenizer.from_pretrained(ref_model_path)

    model.load_state_dict(weights)

    model.save_pretrained(tgt_model_path)
    tokenizer.save_pretrained(tgt_model_path)

    print(f'Model and tokenizer saved at {os.path.abspath(tgt_model_path)}.')


def main():
    """
    Creates a directory for the model and then generates the model itsself
    """

    parser = ArgumentParser()

    # Positional args
    parser.add_argument(
        'ckpt_path',
        type=str,
        help='Path to the .pt checkpoints'
        )
    parser.add_argument(
        'ref_model_path',
        type=str,
        help='Path to the reference OFA model'
    )
    parser.add_argument(
        'tgt_model_path',
        type=str,
        help='Path to save the BiomedGPT model'
    )

    args = parser.parse_args()

    create_model_dir(args.tgt_model_path)
    create_model(str(args.ckpt_path), str(args.ref_model_path), str(args.tgt_model_path))


if __name__ == '__main__':
    main()