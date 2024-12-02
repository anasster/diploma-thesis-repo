"""
Script to postprocess a dataset into an acceptable JSON format, to infer for image captioning
"""


import sys
from argparse import ArgumentParser
from utils import *


def depr_main():
    parser = ArgumentParser()

    parser.add_argument(
        'gastrolab_path',
        type=str,
        help='The path to the gastrolab dataset'
    )
    parser.add_argument(
        'gastro_data_json',
        type=str,
        help='The JSON file to original gastrolab data'
    )
    parser.add_argument(
        'gastro_md_json',
        type=str,
        help='The JSON file storing the metadata'
    )
    parser.add_argument(
        'gastro_freq_json',
        type=str,
        help='The JSON file storing the caption frequencies'
    )
    parser.add_argument(
        'gastro_label_json',
        type=str,
        help='The JSON file storing the label similarities'
    )
    parser.add_argument(
        'gastro_label_txt',
        type=str,
        help='The text file storing the label similarities'
    )
    parser.add_argument(
        'gastro_class_freq',
        type=str,
        help='The JSON file storing class label frequencies for the dataset'
    )
    parser.add_argument(
        'gastro_class_map',
        type=str,
        help='The JSON file storing class map'
    )
    parser.add_argument(
        'gastro_class_hist',
        type=str,
        help='Path to gastrolab histogram for class labels'
    )

    args = parser.parse_args()

    processor = GastrolabPostprocess()

    processor.create_metadata(args.gastrolab_path, args.gastro_data_json, args.gastro_md_json)
    processor.unique_captions(args.gastro_md_json, args.gastro_freq_json)
    processor.find_similar_labels(args.gastro_freq_json, args.gastro_label_json, args.gastro_label_txt)
    processor.merge_labels(args.gastro_md_json, args.gastro_class_map, args.gastro_class_freq)
    processor.make_hist(args.gastro_class_freq, args.gastro_class_hist)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        'dataset',
        type=str,
        help='The dataset to postprocess; \'a\' for Atlas, \'g\' for Gastrolab and \'p\' for Pubmed'
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='The path to the dataset'
    )
    parser.add_argument(
        'data_file',
        type=str,
        help='The path to the JSON file of the dataset'
    )
    parser.add_argument(
        'md_file',
        type=str,
        help='The path to the JSON file storing the postprocessed metadata'
    )
    parser.add_argument(
        'make_hist',
        type=bool,
        help='Enable class histogram generation'
    )
    parser.add_argument(
        'hist_file',
        type=str,
        help='The path to the image file storing the class histograms'
    )
    parser.add_argument(
        'tgt_file_name',
        type=str,
        help='The path to the JSON file for the splits WITHOUT the .json extension'
    )
    parser.add_argument(
        'keep_class_balance',
        type=bool,
        help='Parameter for stratification of the split'
    )
    parser.add_argument(
        'random_seed',
        type=int,
        help='Random seed for the shuffling before the split'
    )

    args = parser.parse_args()

    if args.dataset == 'a':
        processor = AtlasPostprocess()
    elif args.dataset == 'g':
        processor = GastrolabPostprocess()
    elif args.dataset == 'p':
        processor = PubmedPostprocess()
    else:
        raise ValueError('Invalid dataset')
    processor.create_metadata(args.dataset_path, args.data_file, args.md_file, args.make_hist, args.hist_file)
    if args.dataset in ['a', 'g']:
        processor.make_split(args.md_file, args.tgt_file_name, args.random_seed, args.keep_class_balance)
    else:
        processor.make_split(args.md_file, args.tgt_file_name, args.random_seed)
    return 0


if __name__ == '__main__':
    main()
