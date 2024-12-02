"""
Web Scraping: https://www.gastrointestinalatlas.com/english/english.html
"""

import sys
import os
from argparse import ArgumentParser
import logging

from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests

from io import BytesIO
from PIL import Image
import json
import re


URL = 'https://www.gastrointestinalatlas.com/english/english.html'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)


def scrape_website(url=URL, timeout:int=20) -> BeautifulSoup:
    """
    Scrapes the HTML of the requested site.
    Args: 
    url (str): The URL of the site
    timeout (int): Seconds to wait until timeout
    Returns: 
    bs4.BeautifulSoup: The HTML of the site.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()    # Raise error for bad status codes
        logging.info(f'Successfully fetched data from {url}')
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except Exception as e:
        logging.error(f'An error occured: {e}')


def main():
    parser = ArgumentParser(description='Web Scraping Script')

    # Positional arguments
    parser.add_argument(
        'target_dir',
        type=str,
        help='The directory in which images and JSON will be saved.'
    )
    parser.add_argument(
        'json_file',
        type=str,
        default='default.json',
        help='JSON file name for saving images\' caption. Please provide the full relative or absolute path to the file.'
    )

    args = parser.parse_args()

    # Get main page's HTML
    scraped_data = scrape_website(URL)

    # Find a-tags and sanitize them
    a_tags = scraped_data.find_all('a')
    a_tags = a_tags[3:-8]

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
        logging.info(f'Created saving directory: {os.path.abspath(args.target_dir)}')
    else:
        logging.warning(f'Directory {os.path.abspath(args.target_dir)} already exists')
    
    json_object = []
    index = 1
    with open(args.json_file, 'w') as fw:
        json.dump(json_object, fw)
    
    f = open(args.json_file, 'r+')
    data = json.load(f)

    for ind, tag in enumerate(a_tags):
        # Scrape each section's HTML
        link = urljoin(URL, tag['href'])
        tag_html = scrape_website(link)

        # Get label from a-tag text
        try:
            label = tag.text
        except Exception as e:
            logging.exception(f'An error occured: {e}')

        # Find all img tags
        try:
            img_tags = tag_html.find_all('img')
        except Exception as e:
            logging.exception(f'An error occured: {e}')
            continue

        # Download all the images with text
        for img in img_tags:
            try:
                # Get the necessary attributes of the image
                try:
                    text = img['title']
                except:
                    continue
                img_url = urljoin(link, img['src'])
                _, ext = os.path.splitext(os.path.basename(img_url))

                if ext == '.jpg' and text != '':
                    # Create a proper filename
                    name = f'ATLAS_{index}{ext}'

                    # Extract the image's data
                    img_data = BytesIO(requests.get(img_url, timeout=20).content)
                    img_path = re.sub(r'\\', '/', os.path.join(args.target_dir, name))

                    try:
                        with Image.open(img_data) as im:
                            im.save(img_path)
                        index += 1
                        data.append({
                            'name': name,
                            'label': label.strip(),
                            'caption': text.strip()
                        })
                        f.seek(0)
                        json.dump(data, f, indent=2)
                        f.truncate()

                    except Exception as e:
                        logging.exception(f'An error occured: {e}')
            except Exception as e:
                logging.exception(f'An error occured: {e}')
        logging.info(f'Scraping {(ind+1)/len(a_tags)*100:.2f}% complete')
    logging.info(f'Scraping complete. Fetched total {index} items')
    

if __name__ == '__main__':
    main()
