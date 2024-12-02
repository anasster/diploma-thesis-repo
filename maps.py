"""
Script to map state dicts from fairseq-compatible version to transformers-compatible version
"""


import json
import torch
from transformers import OFAModel


# The path to the model in order to map the state dicts
MODEL_PATH = 'PanaceaAI/BiomedGPT-Base-Pretrained'
PT_PATH = 'OFA-base/biomedgpt_base.pt'



def create_map() -> None:
    """
    Maps an OFA model from fairseq version to transformers version in a JSON file
    """
    
    ref_model = OFAModel.from_pretrained(MODEL_PATH)
    ref_dict = ref_model.state_dict()

    fairseq_dict = torch.load(PT_PATH)['model'] # Model we want to change
    
    # Begin comparisons
    json_map = {}
    for rk, rv in ref_dict.items():
        if rk in fairseq_dict.keys():
            for fk, fv in fairseq_dict.items():
                if rv.shape == fv.shape:
                    if rk == fk and bool((rv == fv).all()):
                        json_map[fk] = rk
        else:
            for fk, fv in fairseq_dict.items():
                if rv.shape == fv.shape:
                    if bool((rv == fv).all()):
                        json_map[fk] = rk
    
    with open('key_map.json', 'w') as fo:
        json.dump(json_map, fo, indent=2)
    with open('key_map_inv.json', 'w') as fo:
        json.dump({v: k for k, v in json_map.items()}, fo, indent=2)


def main():
    create_map()


if __name__ == '__main__':
    main()