"""Extract histogram of word count for a caption"""

from argparse import ArgumentParser
import logging
import sys
import json
import seaborn as sns
import matplotlib.pyplot as plt
import ast
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
        'dataset_file',
        type=str,
        help='The JSON file of the dataset'
    )
    parser.add_argument(
        'hist_file',
        type=str,
        help='The PNG file for the histogram'
    )

    args = parser.parse_args()

    data = json.load(open(args.dataset_file, 'r'))

    word_counts = []
    for item in data:
        word_counts.append(item['caption'].split().__len__())
    
    plt.figure()
    sns.histplot(word_counts, bins=100, kde=True)
    plt.xlabel('Number of words')
    plt.ylabel('Frequency')
    plt.title('Word Count Histogram: Train Set')
    plt.savefig(args.hist_file, dpi=350)
    logging.info(f'Histogram created at {sys.argv[2]}')


def alt_main():
    parser = ArgumentParser()

    parser.add_argument(
        'train_file',
        type=str,
        help='The JSON file of the train set'
    )
    parser.add_argument(
        'val_file',
        type=str,
        help='The JSON file of the val set'
    )
    parser.add_argument(
        'test_file',
        type=str,
        help='The JSON file of the test set'
    )
    parser.add_argument(
        'hist_file',
        type=str,
        help='The PNG file for the histogram'
    )

    args = parser.parse_args()

    dataset = []
    dataset_files = sys.argv[1:3]
    for file in dataset_files:
        dataset += json.load(open(file, 'r'))
    
    word_counts = []
    for item in dataset:
        word_counts.append(item['caption'].split().__len__())
    
    plt.figure()
    sns.histplot(word_counts, bins=100, kde=True)
    plt.xlabel('Number of words')
    plt.ylabel('Frequency')
    plt.title('Word Count Histogram: Entire Dataset')
    plt.savefig(args.hist_file, dpi=350)
    logging.info(f'Histogram created at {os.path.abspath(args.hist_file)}')
    

if __name__ == '__main__':
    if len(sys.argv) == 3:
        main()
    else:
        alt_main()
