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

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def add_clip_embeddings_to_csv(image_folder: str, metadata_filepath: str):
    """
    Generates CLIP embeddings for images in a folder and adds them
    to a metadata CSV file, matching by filename (sans extension)
    to the 'release_id' column.

    Args:
        image_folder (str): Path to the folder containing images.
        metadata_filepath (str): Path to the 'releases_metadata.csv' file.
    """
    
    print("Loading CLIP model...")
    # Load the CLIP model
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"CLIP model loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        print("Please ensure CLIP is installed: pip install git+https://github.com/openai/CLIP.git")
        return

    # Check if metadata file exists
    if not os.path.exists(metadata_filepath):
        print(f"Error: Metadata file not found at {metadata_filepath}")
        return

    # Check if image folder exists
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        return

    print(f"Loading metadata from {metadata_filepath}...")
    # Load the metadata CSV
    try:
        df = pd.read_csv(metadata_filepath)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Ensure release_id column is string type for matching filenames
    if 'release_id' not in df.columns:
        print(f"Error: 'release_id' column not found in {metadata_filepath}")
        return
    df['release_id'] = df['release_id'].astype(str)

    # Add 'clip_embedding' column if it doesn't exist
    if 'clip_embedding' not in df.columns:
        df['clip_embedding'] = pd.Series(dtype='object')
        print("Added 'clip_embedding' column to DataFrame.")

    
    print(f"Processing images in {image_folder}...")
    processed_count = 0
    not_found_count = 0
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    # Iterate through files in the image folder
    for filename in os.listdir(image_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in image_extensions:
            # Extract release_id from filename (e.g., "123.jpg" -> "123")
            release_id_str = os.path.splitext(filename)[0]
            image_path = os.path.join(image_folder, filename)

            # Check if this release_id exists in the DataFrame
            if release_id_str not in df['release_id'].values:
                print(f"  - Skipping {filename}: No matching release_id '{release_id_str}' in CSV.")
                not_found_count += 1
                continue
            
            # Check if embedding already exists to avoid reprocessing
            # Find the current embedding value for this release_id
            current_embedding = df.loc[df['release_id'] == release_id_str, 'clip_embedding'].values[0]
            if pd.notna(current_embedding):
                print(f"  = Skipping {filename}: Embedding already exists.")
                continue

            try:
                # Open, preprocess, and move image to device
                image = Image.open(image_path)
                image_input = preprocess(image).unsqueeze(0).to(device)

                # Generate embedding
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                
                # Normalize features (standard practice for CLIP)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Convert embedding to a string representation of a list
                # Detach from graph, move to CPU, convert to numpy, then list
                embedding_list = image_features.cpu().numpy().flatten().tolist()
                embedding_str = str(embedding_list) # Store as string
                
                # Update the DataFrame
                # Use .loc to find the row(s) and set the value
                df.loc[df['release_id'] == release_id_str, 'clip_embedding'] = embedding_str
                print(f"  + Processed {filename} (release_id: {release_id_str})")
                processed_count += 1

            except Exception as e:
                print(f"  - Error processing image {filename}: {e}")

    # Save the updated DataFrame back to the CSV
    try:
        df.to_csv(metadata_filepath, index=False)
        print(f"\nSuccessfully saved updated metadata to {metadata_filepath}")
        print(f"Summary: {processed_count} new embeddings added/updated.")
        if not_found_count > 0:
            print(f"{not_found_count} images were skipped (no matching release_id).")
    except Exception as e:
        print(f"\nError saving updated CSV: {e}")


def ocr_pytesseract(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

def batch_ocr_images(input_folder):
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.splitext(f)[1].lower() in image_exts]
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(input_folder, img_file)
        
        print(f"\nImage: {img_file}")
        
        # Tesseract OCR
        tess_text = ocr_pytesseract(img_path)
        print("[PyTesseract OCR]:")
        print(tess_text if tess_text else "No text found.")


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
            data = {
                'release_id': release.id,
                'cover_url': getattr(release, 'thumb', None)  # thumb is usually the image URL
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

    

    file_path = "C:/cs/portfolio projects/vinyl-classification/.venv/Include/styles.txt"
    styles_df = clean_styles_data(file_path)
    sample_df = pd.DataFrame({'style': ['Drum n Bass','Jazz', 'House']})


    #Extract releases metadata for all styles in the cleaned DataFrame
    metadata_df = extract_metadata_by_styles(sample_df, max_results=20)
    print(metadata_df.head())

    # Optionally, save the metadata to a CSV file
    metadata_df.to_csv("releases_metadata.csv", index=False)


    metadata_path = "C:/cs/portfolio projects/vinyl-classification/releases_metadata.csv"
    download_album_covers(metadata_path, "covers")