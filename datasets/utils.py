"""
Function file for post-processing datasets.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from torchaudio.functional import edit_distance 
import base64
import re
from sklearn.model_selection import StratifiedShuffleSplit
import os
import cv2
from PIL import Image
import shutil
from scipy.ndimage import convolve


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



class DatasetPostprocess:
    r"""Abstract class for postprocessing datasets.\
        Created because pipeline for Atlas & Gastrolab are same in many aspects"""
    
    def __init__(self):
        """Constructor"""
        self._label_map = {}    # Initialize label map as empty dict


    def _sanitize_caption(self, caption) -> str:
        """Sanitize a caption from possible spelling mistakes and/or escape sequences"""
        pass


    def _sanitize_label(self, label) -> str:
        """Sanitize the labels so that 'similar' labels are actually the same"""
        pass


    def create_metadata(self, dataset_path, data_file, md_file, create_hist, hist_file) -> None:
        r"""File for generating a metadata file and sorting classes into labels.\
            Also, if a class has a frequency below 30, sort it into a 'miscellaneous' class."""
        pass


    def make_split(self, metadata_path, tgt_file_name, random_seed, keep_class_dist=True):
        """Create a train-val-test split with 60-20-20 ratio, keeping class balance"""

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata_np = np.array(metadata)

        df = pd.DataFrame(metadata)
        # Set a random seed for the split, and shuffle the data    
        np.random.seed(random_seed)
        np.random.shuffle(metadata)

        n = len(metadata)
        if not keep_class_dist:
            train = metadata[:int(0.6*n)]
            val = metadata[int(0.6*n):int(0.8*n)]
            test = metadata[int(0.8*n):]
        else:
            train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=random_seed)
            for train_inds, test_inds in train_test_split.split(df, df['label']):
                train = metadata_np[train_inds].tolist()
                df_temp = df.loc[test_inds]
            val_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_seed)
            for val_inds, test_inds in val_test_split.split(df_temp, df_temp['label']):
                val = metadata_np[val_inds].tolist()
                test = metadata_np[test_inds].tolist()
        
        try:
            with open(tgt_file_name+'_train.json', 'w') as f1:
                json.dump(train, f1, indent=2)
            with open(tgt_file_name+'_val.json', 'w') as f2:
                json.dump(val, f2, indent=2)
            with open(tgt_file_name+'_test.json', 'w') as f3:
                json.dump(test, f3, indent=2)
            logging.info('Split complete successfully')
        except Exception as e:
            logging.exception(f'An error occured: {e}')


class GastrolabPostprocess(DatasetPostprocess):
    """Class for postprocessing the Gastrolab Dataset"""
    def __init__(self):
        super().__init__()
        self._label_map = {
            'normal': 0,
            'zenker\'s diverticulum': 1,
            'glycogenic acanthosis': 2,
            'oesophageal inlet patch': 3,
            'varices': 4,
            'pseudovarices': 5,
            'oesophageal necrosis': 6,
            'oesophageal diverticulum': 7,
            'oesophagitis': 8,
            'stricture': 9,
            'hiatal hernia': 10,
            'barretts oesophagus': 11,
            'oesophageal tuberculosis': 12,
            'oesophageal papillomatosis': 13,
            'oesophageal cancer': 14,
            'adenocarcinoma': 15,
            'achalasia cardiae': 16,
            'hyperplastic polyp': 17,
            'oesophageal prolapse': 18,
            'mallory weiss tear': 19,
            'ulcer': 20,
            'angiodysplasia': 21,
            'gastritis': 22,
            'lymphoma': 23,
            'linitis plastica': 24,
            'leukemia': 25,
            'metastasis': 26,
            'melanoma': 27,
            'intestinal metaplasia': 28,
            'gist': 29,
            'nsaid': 30,
            'gave': 31,
            'inflammatory polyp': 32,
            'mucosa': 33,
            'duodenitis': 34,
            'papilla vateri': 35,
            'duodenal diverticulum': 36,
            'duodeno jejunal anastomosis': 37,
            'erythema elevatum diutinum': 38,
            'lymphangiectasia': 39,
            'angioectasia': 40,
            'whipples disease': 41,
            'adenoma': 42,
            'gallstone': 43,
            'hepatic cysts': 44,
            'von meyerburg complex': 45,
            'enteroanastomosis': 46,
            'crohn': 47,
            'lymphoid hyperplasia': 48,
            'melanosis coli': 49,
            'colitis': 50,
            'pseudomelanosis': 51,
            'tapeworm': 52,
            'lipoma': 53,
            'diverticulosis': 54,
            'phlebectasia': 55,
            'proctitis': 56,
            'peutz jehgers syndrome': 57,
            'fibromas': 58,
            'haemorrhoids': 59,
            'colon cancer': 60,
            'pouchitis': 61,
            'xanthoma': 62,
            'pinworm': 63,
            'polyposis': 64,
            'miscellaneous': 65
        }
    

    def _sanitize_caption(self, caption):
        sanitized = caption     
        sanitized = re.sub('oesphageal', 'oesophageal', sanitized)
        sanitized = re.sub('sigmoi', 'sigmoid', sanitized)
        sanitized = re.sub('cirkular', 'circular', sanitized)
        sanitized = re.sub('plascica', 'plastica', sanitized)
        sanitized = re.sub('ingestinal', 'intestinal', sanitized)
        sanitized = re.sub('diverticulm', 'diverticulum', sanitized)
        sanitized = re.sub('sigmoidd', 'sigmoid', sanitized)
        sanitized = re.sub(r'[(),:.]', '', sanitized)
        sanitized = re.sub(r'-', ' ', sanitized)

        return sanitized
    

    def _sanitize_label(self, label):
        sanitized = label
        sanitized = re.sub('polyps', 'polyp', sanitized)
        sanitized = re.sub('nsaid s', 'nsaid', sanitized)
        sanitized = re.sub('adenocarcinomas', 'adenocarcinoma', sanitized)
        sanitized = re.sub('ulcers', 'ulcer', sanitized)
        sanitized = re.sub('ulcus', 'ulcer', sanitized)
        sanitized = re.sub('crohn\'s', 'crohn', sanitized)
        sanitized = re.sub('crohns', 'crohn', sanitized)

        return sanitized
    

    def create_metadata(self, dataset_path, data_file, md_file, create_hist, hist_file):       
        # Read the data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Initialize object for storing dataset
        json_object = []

        for item in data:
            name, cap = item.values()

            # Don't use unclassified images (no caption)
            if cap == 'unclassified':
                continue

            sanitized_cap = self._sanitize_caption(cap)
            label = 65
            label_txt = self._sanitize_label(sanitized_cap)
            # Assign a label, if any
            for k, v in self._label_map.items():
                if k in label_txt:
                    label = v
            
            # Encode b64
            img_path = re.sub(r'\\', '/', os.path.join(dataset_path, name))
            with open(img_path, 'rb') as img:
                data = base64.b64encode(img.read()).decode('utf-8')
            
            json_object.append({
                'name': name,
                'caption': sanitized_cap,
                'data': data,
                'label': label
            })

        # Sort out low frequencies
        freq_dict = {}
        for item in json_object:
            lbl = item['label']
            if lbl not in freq_dict.keys():
                freq_dict[lbl] = 1
            else:
                freq_dict[lbl] += 1
        
        # Move to "miscellaneous" whichever class has a frequency below 30
        for i, item in enumerate(json_object):
            lbl = item['label']
            if freq_dict[lbl] < 30:
                json_object[i]['label'] = 65
        
        new_freq_dict = {}
        for item in json_object:
            lbl = item['label']
            if lbl in new_freq_dict.keys():
                new_freq_dict[lbl] += 1
            else:
                new_freq_dict[lbl] = 1
        
        freq_dict = dict(sorted(freq_dict.items()))
        new_freq_dict = dict(sorted(new_freq_dict.items()))

        # Save JSON metadata object
        with open(md_file, 'w') as f:
            json.dump(md_file, f, indent=2)
            logging.info(f'JSON metadata file created at {os.path.abspath(md_file)}')
        
        # Create histogram for classes (unnecessary)
        if create_hist:
            plt.figure()
            plt.subplot(121)
            labels = list(freq_dict.keys())
            frequencies = list(freq_dict.values())
            plt.bar(labels, frequencies)
            plt.title('Original dataset distribution')

            plt.subplot(122)
            labels = list(new_freq_dict.keys())
            frequencies = list(new_freq_dict.values())
            plt.bar(labels, frequencies)
            plt.title('Separated dataset frequencies')
            plt.suptitle('Class Histograms')

            plt.savefig(hist_file)
            logging.info(f'Histogram generated at {os.path.abspath(hist_file)}')


class AtlasPostprocess(DatasetPostprocess):
    """Class for postprocessing the Atlas Dataset"""
    
    def __init__(self):
        super().__init__()
    

    def _sanitize_caption(self, caption):
        sanitized = caption.encode('ascii', 'replace').decode().replace('?', ' ').strip()
        return re.sub(r'[\n\t]', '', ''.join(c for c in sanitized if c.isalnum() or c.isspace()).strip()).strip()
    

    def _sanitize_label(self, label):
        return re.sub(r'\b(I|II|III|IV|V|\d+)\b', '', label).strip()
    

    def create_metadata(self, dataset_path, data_file, md_file, create_hist, hist_file):
        # Load dataset
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Generate class label map
        labels = []
        for item in data:
            if self._sanitize_label(item['label']) in labels or item['label'] in ['Etc, Etc.', 'Important Links', 'Email']:
                continue
            labels.append(self._sanitize_label(item['label']))
        self._label_map = {k: v for k, v in zip(labels, range(len(labels)))}

        # Create metadata
        metadata_object = []
        for item in data:
            name, label, caption = item.values()
            if label in ['Etc, Etc.', 'Important Links', 'Email']:
                break

            sanitized_cap = self._sanitize_caption(caption)
            sanitized_label = self._label_map[self._sanitize_label(label)]

            img_path = re.sub(r'\\', '/', os.path.join(dataset_path, name))
            with open(img_path, 'rb') as img:
                data = base64.b64encode(img.read()).decode('utf-8')
            
            metadata_object.append({
                'name': name,
                'caption': sanitized_cap,
                'label': sanitized_label,
                'data': data
            })

        # Find class frequencies and sort
        freq_dict = {}
        for item in metadata_object:
            lbl = item['label']
            if lbl not in freq_dict.keys():
                freq_dict[lbl] = 1
            else:
                freq_dict[lbl] += 1
        
        for ind, item in enumerate(metadata_object):
            lbl = item['label']
            if freq_dict[lbl] < 30:
                metadata_object[ind]['label'] = len(list(freq_dict.keys()))

        new_freq_dict = {}
        for item in metadata_object:
            lbl = item['label']
            if lbl not in new_freq_dict.keys():
                new_freq_dict[lbl] = 1
            else:
                new_freq_dict[lbl] += 1
        
        with open(md_file, 'w') as f:
            json.dump(metadata_object, f, indent=2)
            logging.info(f'Metadata file created at {os.path.abspath(md_file)}')
        
        if create_hist:
            freq_dict = dict(sorted(freq_dict.items()))
            new_freq_dict = dict(sorted(new_freq_dict.items()))

            plt.figure()
            plt.subplot(121)
            labels = list(freq_dict.keys())
            frequencies = list(freq_dict.values())
            plt.bar(labels, frequencies)
            plt.title('Original dataset distribution')

            plt.subplot(122)
            labels = list(new_freq_dict.keys())
            frequencies = list(new_freq_dict.values())
            plt.bar(labels, frequencies)
            plt.title('Separated dataset frequencies')
            plt.suptitle('Class Histograms')

            plt.savefig(hist_file)
            logging.info(f'Histogram generated at {os.path.abspath(hist_file)}')


class PubmedPostprocess(DatasetPostprocess):
    """Class for postprocessing Pubmed Dataset"""
    def __init__(self):
        super().__init__()
    

    def _sanitize_caption(self, caption):
        sanitized = caption.encode('ascii', 'replace').decode().replace('?', ' ').strip()
        return re.sub(r'[\n\t]', '', ''.join(c for c in sanitized if c.isalnum() or c.isspace()).strip()).strip()
    

    def __filter_hist(self, img) -> float:
        """Filter an image with the sobel filter and return its mean"""
        filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        img_gray = img.convert('L')
        img_numpy = np.array(img_gray)
        img_filt = convolve(img_numpy, filter)
        return np.mean(img_filt)
    
 
    def __find_squares(self, img):
        """Image processing function for finding squares"""
        gray = np.array(img.convert('L'))   # Make grayscale
        padded = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        blurred = cv2.bilateralFilter(padded, 5, 100, 300)    # Blur with bilateral filter
        sharp_filt = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])    # Sharpening kernel
        sharpened = cv2.filter2D(blurred, -1, sharp_filt)   # Sharpen image to make easier edge detection
        _, thresh = cv2.threshold(sharpened, 160, 255, cv2.THRESH_BINARY_INV)   # Make image binary by thresholding
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))    # Shape to perform morphology operations
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)    # Close image
        dilated = cv2.morphologyEx(closed, cv2.MORPH_DILATE, morph_kernel)  # Dilate image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # Find contours of the image

        return len(contours)
    

    def create_metadata(self, dataset_path, data_file, md_file, make_hist, hist_file):
        # Load dataset
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Extract the median of the filtered images' mean values
        hist_means = []
        for item in data:
            if item['caption'] == 'unclassified':
                continue
            img_path = re.sub(r'\\', '/', os.path.join(dataset_path, item['name']))
            hist_means.append(self.__filter_hist(Image.open(img_path)))
        hist_median = np.median(hist_means)

        metadata_object = []
        for item in data:
            name, caption = item.values()
            if caption == 'unclassified': 
                continue

            img_path = re.sub(r'\\', '/', os.path.join(dataset_path, name))
            hist_mean = self.__filter_hist(Image.open(img_path))
            if hist_mean > hist_median:
                num_contours = self.__find_squares(Image.open(img_path))
                if num_contours == 1:
                    with open(img_path, 'rb') as img:
                        data = base64.b64encode(img.read()).decode('utf-8')
                    sanitized_cap = self._sanitize_caption(caption)
                    metadata_object.append({
                        'name': name,
                        'caption': sanitized_cap,
                        'data': data
                    })
        
        # Create metadata file
        with open(md_file, 'w') as f:
            json.dump(metadata_object, f, indent=2)
            logging.info(f'Metadata file created at {os.path.abspath(md_file)}')
        
        if make_hist:
            plt.figure()
            sns.histplot(hist_means, bins=50, kde=True)
            plt.title('Histogram of filtered images\' means')
            plt.savefig(hist_file)
            logging.info(f'Histogram created at {os.path.abspath(hist_file)}')

    
    def make_split(self, metadata_path, tgt_file_name, random_seed):
        """In PUBMED we do not have a class distribution"""
        with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Set a random seed for the split, and shuffle the data    
        np.random.seed(random_seed)
        np.random.shuffle(metadata)

        n = len(metadata)
        train = metadata[:int(0.6*n)]
        val = metadata[int(0.6*n):int(0.8*n)]
        test = metadata[int(0.8*n):]

        try:
            with open(tgt_file_name+'_train.json', 'w') as f1:
                json.dump(train, f1, indent=2)
            with open(tgt_file_name+'_val.json', 'w') as f2:
                json.dump(val, f2, indent=2)
            with open(tgt_file_name+'_test.json', 'w') as f3:
                json.dump(test, f3, indent=2)
            logging.info('Split complete successfully')
        except Exception as e:
            logging.exception(f'An error occured: {e}')
