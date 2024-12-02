"""Module for merging the three datasets, postprocessed into one"""

import json
from argparse import ArgumentParser
import logging
import os
import re
import sys
import ast

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
        'dataset_paths',
        type=list[str],
        help='The dataset directories'
    )
    parser.add_argument(
        'mode',
        type=str,
        help='The split we desire to merge (train, val, test)'
    )

    args = parser.parse_args()

    entire_split = []
    paths = ast.literal_eval(sys.argv[1])
    for dir in paths:
        files = os.listdir(dir)
        for file in files:
            if file.endswith(str(args.mode) + '.json'):
                file_path = re.sub(r'\\', '/', os.path.join(dir, file))
                f = open(file_path, 'r')
                data = json.load(f)
                entire_split += data

    with open(str(args.mode) + '.json', 'w') as f:
        json.dump(entire_split, f, indent=2)
        logging.info('Merge complete')
    return 0


if __name__ == '__main__':
    main()
