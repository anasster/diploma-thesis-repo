"""Script for making JSON files into heatmaps"""

import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging
from argparse import ArgumentParser
import ast
import sys

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
        'json_files',
        type=list[str],
        help='The JSON file paths to the metric files'
    )
    parser.add_argument(
        'image_file',
        type=str,
        help='The file (path) to the image-to-be-saved'
    )

    args = parser.parse_args()

    file_paths = ast.literal_eval(sys.argv[1])
    dicts = []
    for file in file_paths:
        if file.endswith('.json'):
            try:
                f = open(file, 'r')
                data = json.load(f)
                dicts.append(data)
            except Exception as e:
                logging.exception(f'An error occured: {e}')
                    
    arr = []
    for d in dicts:
        arr.append(list(d.values()))
    
    arr = np.array(arr)

    plt.figure()
    sns.heatmap(arr, annot=True, xticklabels=['ROUGE-L', 'METEOR', 'CIDEr'], yticklabels=['Base', 'Medium', 'Tiny'])
    plt.title('Zero-Shot Image Captioning')
    plt.tight_layout()
    plt.savefig(args.image_file)
    logging.info(f'Image generated at {args.image_file}')