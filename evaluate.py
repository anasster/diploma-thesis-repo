"""
Script for zero-shot image captioning
"""


import logging
import sys
import base64
from transformers import OFAModel, OFATokenizer
import json
import torch
from argparse import ArgumentParser
from torchvision import transforms
from PIL import Image
from io import BytesIO
import re
import numpy as np
import os
from time import time

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)

MEAN = [.5, .5, .5]
STD = [.5, .5, .5]
RES = 480
INSTR = ' what does the image describe?'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANSFORM = transforms.Compose([
    lambda image: image.convert('RGB'),
    transforms.Resize((RES, RES), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_data(data_path: str):
    """
    Loads the image data and captions

    params:\n
    data_path: The path to the dataset 
    """

    with open(data_path, 'r') as f:
        try:
            data = json.load(f)
            logging.info('Data loaded successfully')
        except Exception as e: 
            logging.error(f'An error occured during loading the data: {e}')
    return data


def load_model_and_tokenizer(model_path: str, tok_path: str):
    """
    Loads the model and tokenizer and returns them
    """

    model = OFAModel.from_pretrained(model_path)
    tokenizer = OFATokenizer.from_pretrained(tok_path)
    return model, tokenizer


def infer(data_batch, model, tokenizer, num_beams, max_length):
    """
    Returns the generated caption results (in the form of sanitized strings), and the original captions
    """

    images = [Image.open(BytesIO(base64.b64decode(item['data']))) for item in data_batch]
    image_inputs = torch.stack([TRANSFORM(img) for img in images])
    text_inputs = tokenizer([INSTR for _ in data_batch], return_tensors='pt').input_ids

    # Use CUDA
    image_inputs = image_inputs.cuda()
    text_inputs = text_inputs.cuda()
    model = model.cuda()

    # Set to evaluation mode
    model.eval()
    with torch.no_grad():
        gen =  model.generate(
            inputs=text_inputs,
            patch_images=image_inputs,
            num_beams=num_beams, 
            max_length=max_length
            )
    
    captions = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [re.sub(r'[^\w\s]', '', cap).strip() for cap in captions]


def evaluate_captions(refs, hyps, tgt_json_path):
    """
    Calculate Rouge-L, METEOR and CIDEr metrics for generated captions
    """

    scorers = [
        ('Rouge-L', Rouge()),
        ('METEOR', Meteor()), 
        ('CIDEr', Cider())
    ]

    res_json_object = {}
    for sc_name, scorer in scorers:
        res_json_object[sc_name] = scorer.compute_score(refs, hyps)[0]
    
    with open(tgt_json_path, 'w') as f:
        json.dump(res_json_object, f, indent=2)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        'data_path',
        type=str,
        help='The path to the dataset'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='The path to the model'
    )
    parser.add_argument(
        'tok_path',
        type=str,
        help='The path to the tokenizer'
    )
    parser.add_argument(
        'batch_size',
        type=int,
        help='Batch size for inference'
    )
    parser.add_argument(
        'save_path',
        type=str,
        help='The path to save the evaluation metrics'
    )
    parser.add_argument(
        'res_json_filename',
        type=str,
        help='The name of the JSON file to store results in'
    )
    parser.add_argument(
        'num_beams',
        type=int,
        help='Number of beams'
    )
    parser.add_argument(
        'max_tgt_length',
        type=int,
        help='Maximum length of generated captions'
    )
    parser.add_argument(
        '--shuffle',
        type=bool,
        help='Indicate whether to shuffle the dataset before inference'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        help='Seed for random operations'
    )

    args = parser.parse_args()

    # Start time
    s = time()

    # Load data, model and tokenizer
    data = load_data(args.data_path)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tok_path)
    logging.info(f'Loaded model and tokenizer @ {time()-s} s')

    # Set random seed if we want to infer with shuffled batches
    if args.shuffle:
        np.random.seed(args.random_seed)

        # Shuffle the batch to reduce bias
        np.random.shuffle(data)
    
    gt_caps = {}    # Ground truths
    hyp_caps = {}    # Hypothesis captions

    num_batches = len(data)//args.batch_size
    current_batch = 1

    # Create a directory to save results
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        logging.info(f'Directory for results {args.save_path} created')
    else:
        logging.warning(f'Path {args.save_path} already exists')

    try:
        for i in range(0, len(data), args.batch_size):
            data_batch = data[i:i+args.batch_size] if i+args.batch_size <= len(data) else data[i:len(data)]
            
            # Make inference
            gen_caps = infer(data_batch, model, tokenizer, args.num_beams, args.max_tgt_length)

            # Store results
            for item in data_batch:
                gt_caps[item['name']] = [item['caption']]
            for item, cap in zip(data_batch, gen_caps):
                hyp_caps[item['name']] = [cap]
            
            logging.info(f'Inference----------{current_batch}/{num_batches} batches')
            current_batch += 1
        logging.info(f'Inference complete @ {time()-s} s')
    except Exception as e:
        logging.exception(f'An error occured: {e}')
    

    evaluate_captions(gt_caps, hyp_caps, re.sub(r'\\', '/', os.path.join(args.save_path, args.res_json_filename)))
    logging.info(f'Results saved at {os.path.abspath(args.save_path)}')


if __name__ == '__main__':
    main()
