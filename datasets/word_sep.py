"""Split the dataset according to word count; For test set, split into 'short' (words<7) and 'long'"""

import json
import logging
import sys
from argparse import ArgumentParser
import numpy as np
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        'data_file',
        type=str,
        help='The JSON file of the dataset to be separated'
    )
    
    args = parser.parse_args()

    data = json.load(open(args.data_file, 'r'))
    words = np.array([len(item['caption'].split()) for item in data])
    short, long = [], []
    for item in data:
        if len(item['caption'].split()) < np.median(words):
            short.append(item)
        else:
            long.append(item)
    
    # Save two different files
    name, ext = os.path.splitext(args.data_file)
    try:
        json.dump(short, open(name+'_short'+ext, 'w'), indent=2)
        json.dump(long, open(name+'_long'+ext, 'w'), indent=2)
        logging.info(f'Dataset separated')
    except Exception as e:
        logging.exception(f'An error occured: {e}')


if __name__ == '__main__':
    main()
