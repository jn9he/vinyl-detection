import pandas as pd
import re
import os
import discogs_client
import time
import webbrowser
from dotenv import load_dotenv
from tqdm import tqdm
import requests

import pytesseract
from PIL import Image
from google.cloud import vision

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import clip

import cv2
from glob import glob
import easyocr

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def batch_clip_embeddings_to_dataframe(folder_path):
    # Load model and processor once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    allowed_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in allowed_exts]
    
    records = []
    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
            embedding_np = embedding.cpu().numpy()[0]
            embedding_np = embedding_np / np.linalg.norm(embedding_np)
            record = {'filename': fname}
            # Add each embedding dimension as a separate column (dim_0, dim_1, ..., dim_511)
            for i, val in enumerate(embedding_np):
                record[f'dim_{i}'] = val
            records.append(record)
        except Exception as e:
            print(f"Error processing '{fname}': {e}")
    
    df = pd.DataFrame(records)
    return df

#clip embedding for an individual png
def print_clip_embedding(image_path):
    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Forward pass to get embedding
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding_np = embedding.cpu().numpy()[0]
    embedding_np = embedding_np / np.linalg.norm(embedding_np)  # Normalize

    print(f"CLIP embedding for '{image_path}':")
    print(embedding_np)
    
#rewrite this function to preprocess the image using OpenCV before passing it to pytesseract - not sure what transformations are done by easyocr?
def ocr_with_easyocr(image_path, lang_list=['en']):
    reader = easyocr.Reader(lang_list, gpu=False)
    results = reader.readtext(image_path, detail=0)
    return ' '.join(results)

def ocr_with_tesseract(image_path, lang='eng'):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang=lang)

def batch_ocr(image_folder, output_csv="ocr_results.csv"):
    import pandas as pd
    results = []
    image_paths = glob(os.path.join(image_folder, '*.*g'))  # jpg, png, etc
    
    # Uncomment only the engines you want to run:
    # Set GOOGLE_APPLICATION_CREDENTIALS env variable for Google Vision
    
    for path in image_paths:
        res = {'filename': os.path.basename(path)}
        print(f"[Processing] {path}")
        # EasyOCR
        try:
            res['easyocr'] = ocr_with_easyocr(path, lang_list=['en'])
        except Exception as e:
            print(f"EasyOCR failed for {path}: {e}")
            res['easyocr'] = ''
        results.append(res)
        # Pytesseract
        # try:
        #     res['tesseract'] = ocr_with_tesseract(path, lang='eng')
        # except Exception as e:
        #     print(f"Tesseract failed for {path}: {e}")
        #     res['tesseract'] = ''
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"OCR results saved to {output_csv}")

def download_album_covers(metadata_path, output_dir="album_covers"):
    """
    Download album cover images from a metadata file
    (expects 'release_id' and 'cover_url' columns).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)  # Or change to pd.read_parquet if needed
    
    # Use a session for connection pooling and optional retries
    session = requests.Session()
    headers = {
        'User-Agent': 'vibeVinyl/1.0 (+https://github.com/yourname/vibeVinyl)'
    }

    saved = 0
    skipped = 0
    failed = 0

    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        cover_url = row.get('cover_url')
        release_id = str(row.get('release_id'))

        # Normalize and skip missing or invalid URLs
        if not cover_url or str(cover_url).strip().lower() in ['nan', 'none']:
            skipped += 1
            continue

        cover_url = str(cover_url).strip()

        try:
            response = session.get(cover_url, headers=headers, timeout=15, allow_redirects=True)
            status = response.status_code
            if not response.ok or not response.content:
                print(f"Skipping {cover_url} (status={status}, empty content)")
                failed += 1
                continue

            # Determine file extension from content-type header
            content_type = response.headers.get('Content-Type', '').lower()
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            else:
                # default to jpg if unknown
                ext = '.jpg'

            fname = os.path.join(output_dir, f"{release_id}{ext}")
            with open(fname, 'wb') as f:
                f.write(response.content)
            saved += 1
        except Exception as e:
            print(f"Error downloading {cover_url} for release {release_id}: {e}")
            failed += 1

    print(f"Download complete. Images saved to {output_dir} (saved={saved}, skipped={skipped}, failed={failed})")

    print(f"Download complete. Images saved to {output_dir}")

def create_discogs_client():
    """
    Create and return a configured discogs_client.Client.

    Looks for credentials in the following environment variables (in order):
      - DISCOGS_USER_TOKEN or DISCOGS_TOKEN : a Discogs personal access token
      - DISCOGS_CONSUMER_KEY and DISCOGS_CONSUMER_SECRET : OAuth consumer creds

    Raises:
        EnvironmentError: if no credentials are found.
    """
    user_token = os.getenv('DISCOGS_USER_TOKEN') or os.getenv('DISCOGS_TOKEN')
    consumer_key = os.getenv('DISCOGS_CONSUMER_KEY')
    consumer_secret = os.getenv('DISCOGS_CONSUMER_SECRET')

    # Provide a descriptive user agent (app name/version and contact URL is recommended)
    user_agent = 'vibeVinyl/1.0 +https://github.com/yourname/vibeVinyl'

    if user_token:
        return discogs_client.Client(user_agent, user_token=user_token)
    elif consumer_key and consumer_secret:
        return discogs_client.Client(user_agent, consumer_key=consumer_key, consumer_secret=consumer_secret)
    else:
        raise EnvironmentError(
            'No Discogs credentials found. Set DISCOGS_USER_TOKEN (or DISCOGS_TOKEN) '
            'or DISCOGS_CONSUMER_KEY and DISCOGS_CONSUMER_SECRET as environment variables.'
        )

def get_releases_by_style(style, max_results=50):
    releases = []
    per_page = 50 if max_results > 50 else max_results
    result_pages = d.search(style=style, type='release', per_page=per_page)

    results_yielded = 0
    current_page = 1
    while results_yielded < max_results:
        page_results = result_pages.page(current_page)
        if not page_results:
            break

        for release in page_results:
            # Prefer high quality cover image if available
            cover_url = getattr(release, 'cover_image', None)
            if not cover_url:
                cover_url = getattr(release, 'thumb', None)  # fallback to thumbnail
            
            data = {
                'release_id': release.id,
                'cover_url': cover_url
            }
            releases.append(data)
            results_yielded += 1
            if results_yielded >= max_results:
                break
        current_page += 1
    return releases


def extract_metadata_by_styles(df, max_results=20):
    all_releases = []
    for _, row in df.iterrows():
        style = row['style']
        print(f"Querying style: {style}")
        releases = get_releases_by_style(style, max_results=max_results)
        # Skip releases that don't have a cover URL (cover_url may be None or empty)
        releases = [r for r in releases if r.get('cover_url')]
        all_releases.extend(releases)
        time.sleep(1.2)  # To respect Discogs' API rate limit
    return pd.DataFrame(all_releases)

def clean_styles_data(input_file, output_file='cleaned_styles.csv'):
    """
    Clean the Discogs styles data by extracting just the style names
    
    Args:
        input_file: Path to the input text file with style data
        output_file: Path for the cleaned CSV output
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame with just style names
    """
    styles = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and the header line
            if not line or line.startswith('Style'):
                continue
            
            # Remove all numbers and commas, keep only text
            style_name = re.sub(r'[\d,]+', '', line).strip()
            
            if style_name:
                styles.append(style_name)
    
    # Create DataFrame
    df = pd.DataFrame({'style': styles})

    # Save to CSV
    df.to_csv(output_file, index=False)

    return df

if __name__ == "__main__":
    load_dotenv()
    d = create_discogs_client()

    # file_path = "/Users/joshnghe/Desktop/Code/personal/portfolio projects/vinyl-detection/.venv/include/vinyl-detection/styles.txt"
    # styles_df = clean_styles_data(file_path)
    # sample_df = pd.DataFrame({'style': ['Drum n Bass','Jazz', 'House']})

    # #Extract releases metadata for all styles in the cleaned DataFrame
    # metadata_df = extract_metadata_by_styles(sample_df, max_results=20)
    # print(metadata_df.head())

    # # Optionally, save the metadata to a CSV file
    # metadata_df.to_csv("releases_metadata.csv", index=False)

    # #extract album covers using the urls insid ethe releaes metadata file.
    # metadata_path = "/Users/joshnghe/Desktop/Code/personal/portfolio projects/vinyl-detection/releases_metadata.csv"
    # download_album_covers(metadata_path, "covers")

    #perform batch ocr embedding, and pipe the embeddings to a new column in the metadata csv file.
    #pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.5.1/bin/tesseract'
    #batch_ocr("/Users/joshnghe/Desktop/Code/personal/portfolio projects/vinyl-detection/covers")
    #perform batch clip embedding, and pipe the embeddings to a new column in the metadata csv file.


    #batch clip embedding
    #print_clip_embedding("C:/cs/portfolio projects/vinyl-classification/covers/3960.jpg")
    df = batch_clip_embeddings_to_dataframe("covers")
    print(df.head())
    print(len(df))