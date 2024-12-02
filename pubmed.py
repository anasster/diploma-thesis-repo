"""
Web Scraping: https://www.ncbi.nlm.nih.gov/pmc/
"""

import os
import logging
import sys
import re

from argparse import ArgumentParser
import time
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urljoin
import json

from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


# Global constants for URL, user-agent headers and keyword for the scraping process
URL = 'https://www.ncbi.nlm.nih.gov/pmc/'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
}
SEARCH_KW = 'capsule endoscopy'


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console
    ]
)


def main():
    """
    Main function for downloading images from PUBMED and saving them in 
    a JSON file with their corresponding text.
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
    parser.add_argument(
        'search_keyword',
        type=str,
        default=SEARCH_KW,
        help='Keyword to search for in PUBMED site.'
    )

    # Optional arguments
    parser.add_argument(
        '--timeout',
        type=int,
        default=15,
        help='Timeout for request.'
    )

    args = parser.parse_args()

    # Open a webdriver and the site
    driver = Chrome()
    driver.get(URL)

    # Search for a KW
    search_bar = driver.find_element(By.ID, 'pmc-search')
    search_bar.send_keys(args.search_keyword)
    search_bar.send_keys(Keys.RETURN)

    # At the page, make 100 elements appear per page to save time
    img_btn = driver.find_element(By.CLASS_NAME, 'seemore')
    img_btn.click()
    display_settings = driver.find_element(By.CLASS_NAME, 'left.display_settings.content_header.jig-ncbipopper')
    display_settings.click()
    time.sleep(3)
    num_el = driver.find_element(By.ID, 'ps100') # Find the 100 elements per page button
    num_el.click()

    # Create a download directory to save the images
    if not os.path.exists(args.target_dir):
        target_dir = os.makedirs(args.target_dir)
        logging.info(f'Download directory successfully created at {os.path.abspath(args.target_dir)}')
    else:
        logging.warning(f'Directory {os.path.abspath(args.target_dir)} already exists.')
    
    json_object = []    # JSON object to save the scraped data
    index = 1   # Index to name each file

    # Download sequence
    while True:
        # Find img tags and texts
        elements = driver.find_elements(By.CLASS_NAME, 'rslt')
        img_tags = []
        texts = []
        for element in elements:
            try:
                img_tags.append(element.find_element(By.TAG_NAME, 'a').find_element(By.TAG_NAME, 'img').get_attribute('src-large'))
                try:
                    texts.append(element.find_element(By.CLASS_NAME, 'supp').text.strip())
                except:
                    texts.append('unclassified')
            except:
                continue
        
        # Download what was found
        for img, txt in zip(img_tags, texts):
            img_url = urljoin(driver.current_url, img)
            _, ext = os.path.splitext(img_url)

            if ext == '.jpg':
                try:
                    img_data = BytesIO(requests.get(img_url, headers=HEADERS).content)
                    filepath = os.path.join(args.target_dir, f'PUBMED_{index}{ext}')
                    with Image.open(img_data) as f:
                        f.save(filepath)
                    json_object.append({
                        'name': os.path.basename(filepath),
                        'caption': txt
                    })
                    index += 1
                except Exception as e:
                    logging.error(f'An error occured: {e}')
        
        try:
            bot_elmt = driver.find_element(By.CLASS_NAME, 'title_and_pager.bottom')
            next_page_btn = bot_elmt.find_element(By.CLASS_NAME, 'active.page_link.next')
            next_page_btn.click()
            logging.info('Moving on to next page...')
        except:
            logging.info(f'Image downloads complete. Downloaded {index} images.')
            break
    
    # Create the JSON file
    with open(re.sub(r'\\', '/', os.path.join(args.target_dir, args.json_filename)), 'w') as fo:
        json.dump(json_object, fo, indent=2)
    

if __name__ == '__main__':
    main()