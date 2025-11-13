from flask import Flask, request, jsonify, send_from_directory
import os
import re
import base64
from datetime import datetime
import io
import contextlib
import sys

# Ensure the project's .venv/Include directory is on the import path so we can import extract_images
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '.venv', 'Include'))
import extract_images

app = Flask(__name__, static_folder='.', template_folder='.')

SNAPSHOT_DIR = os.path.join(os.getcwd(), 'snapshots_web')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

DATA_URI_RE = re.compile(r'data:(?P<mime>[-\w/+.]+);base64,(?P<data>.+)')

@app.route('/')
def index():
    # Serve the static HTML UI located in repository root
    # Serve the web_ui.html from the same folder as this server file
    webapp_dir = os.path.dirname(__file__)
    return send_from_directory(webapp_dir, 'web_ui.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        payload = request.get_json(force=True)
        data_url = payload.get('image')
        if not data_url:
            return jsonify({'error': 'Missing image data'}), 400

        m = DATA_URI_RE.match(data_url)
        if not m:
            return jsonify({'error': 'Invalid data URI'}), 400

        mime = m.group('mime')
        b64 = m.group('data')
        # Choose extension from mime
        if mime == 'image/jpeg' or mime == 'image/jpg':
            ext = '.jpg'
        elif mime == 'image/png':
            ext = '.png'
        else:
            # default
            ext = '.jpg'

        binary = base64.b64decode(b64)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'snapshot_{ts}{ext}'
        out_path = os.path.join(SNAPSHOT_DIR, filename)

        with open(out_path, 'wb') as f:
            f.write(binary)

        # Perform OCR using the helper from extract_images.py
        try:
            ocr_text = extract_images.ocr_with_easyocr(out_path)
        except Exception as e:
            ocr_text = f"OCR error: {e}"

        # Capture printed CLIP embedding output from helper
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                extract_images.print_clip_embedding(out_path)
            clip_text = buf.getvalue().strip()
        except Exception as e:
            clip_text = f"CLIP error: {e}"

        return jsonify({'ok': True, 'filename': filename, 'ocr': ocr_text, 'clip': clip_text}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/snapshots/<path:fname>')
def serve_snapshot(fname):
    return send_from_directory(SNAPSHOT_DIR, fname)

if __name__ == '__main__':
    # Run development server
    print('Serving web UI at http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=5000, debug=True)
