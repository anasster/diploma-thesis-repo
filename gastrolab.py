"""
Web Scraping: https://gastrolab.net
"""


import os
import sys
from argparse import ArgumentParser
import logging

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
import json


# Global constant for site-to-be-scraped URL
URL = 'https://gastrolab.net/'


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)


def scrape_website(url=URL, timeout:int=1) -> BeautifulSoup:
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
    """
    The main function that performs the web-scraping and saves the images,
    as well as a JSON file with the corresponding text on target directory.
    """
    parser = ArgumentParser(description='Web Scraping Script')

    # Positional arguments
    parser.add_argument(
        'target_dir',
        type=str,
        help='The directory in which images and JSON will be saved.'
    )
    parser.add_argument(
        'json_filename',
        type=str,
        default='default.json',
        help='JSON file name for saving images\' caption. Please provide ONLY the name of the file.'
    )

    # Optional arguments
    parser.add_argument(
        '--timeout',
        type=int,
        help='Timeout for request.'
    )

    args = parser.parse_args()

    # Call the scraper
    scraped_data = scrape_website(URL, args.timeout)

    # Create a download directory to save the images
    try:
        target_dir = os.makedirs(args.target_dir, exist_ok=True)
        logging.info('Download directory successfully created.')
    except FileExistsError as e:
        logging.error(f'An error occured: {e}')
    
    # Sanitize the HTML, as we want to keep a piece of it
    a_tags = scraped_data.find_all('a')[12:]    # Keep only the a-tags
    a_tags = [tag for tag in a_tags if 'NEW! COLLAGEs' not in tag.text]

    json_object = []    # JSON object to save the scraped data
    index = 1   # Index to name each file

    # Begin download sequence
    for ind, tag in enumerate(a_tags):
        # Get the text, mutual for all images and sanitize it
        text = tag.get_text(strip=True)
        text = text.replace('NEW!SLIDES:', '').replace('NEW!', '').replace('IMAGE:', '').replace('(Slides)', '')

        # Scrape each a-tag's HTML
        tag_url = urljoin(URL, tag['href'])
        tag_html = scrape_website(tag_url, timeout=args.timeout)
        
        # Find the img and h3 tags
        try:
            tag_elements = tag_html.find_all(['img', 'h3'])
            for element in tag_elements:
                # Stop downloading at h3-tag
                if element.name == 'h3':
                    break
                else:
                    img_url = urljoin(tag_url, element['src'])
                    try:
                        img = Image.open(BytesIO(requests.get(img_url).content)).convert('RGB')
                        _, ext = os.path.splitext(img_url)
                        if ext == '.jpg':
                            img_path = os.path.join(args.target_dir, f'GASTROLAB_{index}{ext}')
                            img.save(img_path)
                            json_object.append({
                                'name': os.path.basename(img_path),
                                'caption': text.strip()
                                })
                            index += 1
                    except Exception as e:
                        logging.error(f'Could not download from {img_url} due to {e}.')
        except Exception as e:
            logging.error(f'An error occured: {e}')            
        logging.info(f'Downloads are {(ind+1)/len(a_tags)*100:.2f}% complete.')
    logging.info(f'Downloads complete. Downloaded {index} objects.')

    # Save the images' text as a JSON file
    with open(args.target_dir + '/' + args.json_filename, 'w') as fo:
        json.dump(json_object, fo, indent=2)
        logging.info('JSON file successfully created.')
    
    logging.info('Dataset created.')


if __name__=='__main__':
    main()