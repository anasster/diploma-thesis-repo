"""
Script to transform and save transformers compatible checkpoints for BiomedGPT
"""


import torch
from collections import OrderedDict
import json
import logging 
import os
import sys
from argparse import ArgumentParser
from transformers import OFAModel


with open('key_map.json', 'r') as f:
    KEY_MAP = json.load(f)

with open('key_map_inv.json', 'r') as f:
    KEY_MAP_INV = json.load(f)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ]
)


def generate_checkpoint(model: str, ckpt_path: str, save_path: str) -> None:
    """
    Generates new transformers compatible checkpoints

    params:\n
    model(str): Tiny, medium or base version\n
    ckpt_path(str): Path to find fairseq checkpoint\n
    save_path(str): Path to save transformers checkpoint
    """

    models = ['tiny', 'medium', 'base']
    if model not in models:
        raise ValueError('Model param must be "tiny", "medium" or "base".')
    
    if model == 'tiny':
        fair_dict = torch.load(ckpt_path)['model']
        del fair_dict['encoder.version']
        del fair_dict['decoder.version']
        ref_dict = OFAModel.from_pretrained('./OFA-tiny').state_dict()

        new_dict = OrderedDict((k, fair_dict[KEY_MAP_INV[k]]) for k in ref_dict.keys())
        new_dict['decoder.image_position_idx'] = new_dict['decoder.image_position_idx'][:-1]
        torch.save(new_dict, save_path)
    
    if model == 'medium':
        ref_dict = torch.load(ckpt_path)['model']
        del ref_dict['encoder.version']
        del ref_dict['decoder.version']

        new_dict = OrderedDict((KEY_MAP.get(k), v) for k, v in ref_dict.items())
        torch.save(new_dict, save_path)
    
    if model == 'base':
        ref_dict = torch.load(ckpt_path)['model']
        del ref_dict['encoder.version']
        del ref_dict['decoder.version']

        new_dict = OrderedDict((KEY_MAP.get(k), v) for k, v in ref_dict.items())
        try:
            os.makedirs(save_path, exist_ok=False)
            logging.info(f'Save path created')
        except OSError as e:
            logging.info(f'An error occured: {e}')

        torch.save(new_dict, save_path)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        'model',
        type=str,
        help='The type of the model\'s checkpoint (tiny, medium, base)'
    )
    parser.add_argument(
        'ckpt_path',
        type=str,
        help='The path to the fairseq checkpoint that we wan to convert'
    )
    parser.add_argument(
        'save_path',
        type=str,
        help='The path to save the transformers checkpoint'
    )
    
    args = parser.parse_args()

    try:
        generate_checkpoint(args.model, args.ckpt_path, args.save_path)
        logging.info('Checkpoint generated successfully.')
    except Exception as e:
        logging.exception(f'An error occured: {e}')


if __name__ == '__main__':
    main()