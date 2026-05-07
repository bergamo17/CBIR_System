import faiss
import clip
import torch
import numpy as np
import json
import base64
import os
import re #changes
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import io
import hashlib
import psycopg2
from psycopg2.extras import Json
import cv2
from mobile_sam import SamPredictor, sam_model_registry
import time
from dotenv import load_dotenv


app = Flask(__name__, static_folder='static')
CORS(app)

load_dotenv()

FEATURE_BANK_DIR = 'Feature_Bank_New'
INDEX_PATH = 'Feature_Bank_New/faiss.index'
DATASET_DIR = 'Dataset_new'
KAGGLE_PREFIX = '/kaggle/input/datasets/arielsinaga/poc-taco/Dataset_new/'
LOCAL_DATASET = 'Dataset_new/'
# changes
OFFICIAL_IMAGES_DIR = 'Official Image'



model = None
preprocess = None
device = None
faiss_index = None
metadata_list = None
sam_model = None
mask_generator = None

# changes
official_image_index = {}

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_image_hash(pil_image):
    t0 = time.time()
    print("[get_image_hash] Start processing...")

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="PNG")

    print(f"[get_image_hash] End process ({(time.time() - t0) * 1000:.1f} ms)")
    return hashlib.sha256(img_byte_arr.getvalue()).hexdigest()

def extract_base_code(text: str):
    t0 = time.time()
    print("[extract_base_code] Starting process...")
    text = text.upper()
    match = re.search(r'([A-Z]+)[\s-]?(\d+)', text)
    result = f"{match.group(1)} {match.group(2)}" if match else None
 
    print(f"[extract_base_code] End process ({(time.time() - t0) * 1000:.1f} ms)")
    return result

def build_official_image_index():
    global official_image_index
    t0 = time.time()
    print("[build_official_image_index] Starting process...")

    official_image_index = {}

    official_dir = Path(OFFICIAL_IMAGES_DIR)

    for f in official_dir.iterdir():
        stem = f.stem  # contoh: "TH 352 H - Brown"

        base_code = extract_base_code(stem)

        if base_code:
            # simpan hanya 1 image per base code
            if base_code not in official_image_index:
                official_image_index[base_code] = str(f)

    print("[CBIR] Official index:", official_image_index)
    print(f"[build_official_image_index] End process ({(time.time() - t0) * 1000:.1f}ms)")

def get_official_image_key(image_name: str):
    t0 = time.time()
    print("[get_official_image_key] Starting process...")
    stem = Path(image_name).stem

    # Remove (1), (2), etc
    stem = re.sub(r'\s*\(\d+\)\s*$', '', stem)

    # Replace ALL hyphens with space
    stem = stem.replace('-', ' ')

    # Normalize multiple spaces
    stem = re.sub(r'\s+', ' ', stem)

    print(f"[get_official_image_key] End process ({(time.time() - t0) * 1000:.1f}ms)")

    return stem.upper().strip()

def normalize_class_name(class_name: str):
    t0 = time.time()
    print("[normalize_class_name] Start processing...")
    class_name = class_name.replace('-', ' ')
    class_name = re.sub(r'\s+', ' ', class_name)

    print(f"[normalize_class_name] End process ({(time.time() - t0) * 1000:.1f}ms)")
    return class_name.upper().strip()


def find_official_image_by_class(class_name: str):
    t0 = time.time()
    print("[find_official_image_by_class] Start Processing...")

    base_code = extract_base_code(class_name)

    print(f"[find_official_image_by_class] End process ({(time.time() - t0) * 1000:.1f}ms)")
    return official_image_index.get(base_code)

def get_result_image_b64(result: dict) -> str:
    t0 = time.time()
    print("[get_result_image_b64] Start Processing...")
    class_name = result.get('class', '')

    # 1. Coba official image
    official_path = find_official_image_by_class(class_name)
    if official_path:
        b64 = image_to_base64(official_path)
        if b64:
            return b64

    print(f"[get_result_image_b64] End process ({(time.time() - t0) * 1000:.1f}ms)")
    # 2. Fallback ke dataset
    return image_to_base64(result.get('image_path', ''))

#___________________________________________________________

def l2_normalization(vector):
    t0 = time.time()
    print("[l2_normalization] Start Processing...")
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector
    print(f"[l2_normalization] End process ({(time.time() - t0) * 1000:.1f}ms)")
    return vector / magnitude

def load_assets():
    global model, preprocess, device, faiss_index, metadata_list, sam_model, mask_generator
    t0 = time.time()
    print("[load_assets] Start Processing...")
 
    print("[CBIR] Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/16', device=device)
    model.eval()
    print(f"[CBIR] CLIP loaded on: {device}")
 
    print("[CBIR] Loading FAISS index...")
    faiss_index = faiss.read_index(INDEX_PATH)
    print(f"[CBIR] FAISS index: {faiss_index.ntotal} vectors")
 
    print("[CBIR] Loading metadata...")
    metadata_path = Path(FEATURE_BANK_DIR) / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    print(f"[CBIR] Metadata: {len(metadata_list)} entries")
    print("[CBIR] Building official image index...")
    build_official_image_index()

    print("[CBIR] Loading MobileSAM model...")
    sam_checkpoint = "mobile_sam.pt"
    sam_model = sam_model_registry['vit_t'](checkpoint=sam_checkpoint)
    sam_model.to(device)
    mask_generator = SamPredictor(sam_model)
    print(f"[CBIR] MobileSAM loaded on: {device}")
    
    print("[CBIR] All assets loaded. Ready!")
    print(f"[load_assets] End process ({(time.time() - t0) * 1000:.1f}ms)")


def remove_background(pil_image: Image.Image) -> Image.Image:
    t0 = time.time()
    print("[remove_background] Start Processing...")
    try:
        img_rgb = np.array(pil_image.convert('RGB'))
        h, w = img_rgb.shape[:2]

        mask_generator.set_image(img_rgb)
        center_point = np.array([[w // 2, h // 2]])
        center_label = np.array([1])  # 1 = foreground

        masks, scores, _ = mask_generator.predict(
            point_coords=center_point,
            point_labels=center_label,
            multimask_output=True  # generate 3 kandidat mask
        )

        # Pilih mask dengan confidence score tertinggi
        best_mask = masks[np.argmax(scores)]

        # Terapkan mask → background transparan
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = (best_mask * 255).astype(np.uint8)

        result = Image.fromarray(rgba)
        print(f"[remove_background] End process ({(time.time() - t0) * 1000:.1f} ms) → background removed")
        return result

    except Exception as e:
        print(f"[SAM] Error: {e}, skip background removal")
        print(f"[remove_background] End process ({(time.time() - t0) * 1000:.1f}ms)")
        return pil_image
    

def extract_query_vector(pil_image):
    t0 = time.time()
    print("[extract_query_vector] Start Processing...")
    pil_image = remove_background(pil_image)
    img = pil_image.convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(img_tensor)
    raw_vector = feature.squeeze().cpu().numpy().astype('float32')
    print(f"[extract_query_vector] End process ({(time.time() - t0) * 1000:.1f}ms)")
    return l2_normalization(raw_vector)

def similarity_search(query_vector, top_k=5):
    t0 = time.time()
    print("[similarity_search] Start Processing...")
    query_2d = query_vector.reshape(1, -1)
    distances, indices = faiss_index.search(query_2d, top_k)
    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        meta = metadata_list[idx]
        results.append({
            'rank':       rank + 1,
            'similarity_score':   round(float(dist), 4),
            'faiss_index': int(idx),
            'class':      meta['class'],
            'image_path': meta['image_path'],
            'image_name': meta['image_name'],
        })
    print(f"[similarity_search] End process ({(time.time() - t0) * 1000:.1f}ms)")
    return results

def image_to_base64(image_path):
    """Convert a dataset image to base64 for embedding in JSON response."""
    t0 = time.time()
    print("[image_to_base64] Start Processing...")
    try:
        if image_path.startswith(KAGGLE_PREFIX):
            image_path = image_path.replace(KAGGLE_PREFIX, LOCAL_DATASET)

        full_path = Path(image_path)
        if full_path.exists():
            with Image.open(full_path) as img:
                img.thumbnail((300, 300))
                buffer = io.BytesIO()
                fmt  = img.format or 'PNG'
                mime = 'image/png' if fmt == 'PNG' else 'image/jpeg'
                if fmt == 'PNG':
                    img.save(buffer, format='PNG', optimize=True)
                else:
                    img.convert('RGB').save(buffer, format='JPEG', quality=85)
                b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                print(f"[image_to_base64] End process ({(time.time() - t0) * 1000:.1f}ms)")
                return f"data:{mime};base64,{b64}"
        else:
            print(f"[WARN] File tidak ditemukan: {full_path.resolve()}")
            print(f"[image_to_base64] End process ({(time.time() - t0) * 1000:.1f}ms)")
    except Exception as e:
        print(f"[WARN] Cannot load image {image_path}: {e}")
        print(f"[image_to_base64] End process ({(time.time() - t0) * 1000:.1f}ms)")
    return None


# ROUTES

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    top_k = int(request.form.get('top_k', 5))

    try:
        pil_image = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400
    
    query_buffer = io.BytesIO()
    pil_image.convert('RGB').thumbnail((400, 400))
    pil_image.convert('RGB').save(query_buffer, format='JPEG', quality=85)
    query_b64 = 'data:image/jpeg;base64,' + base64.b64encode(query_buffer.getvalue()).decode('utf-8')

    img_hash = get_image_hash(pil_image)

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT refined_vector FROM image_history WHERE image_hash = %s", (img_hash,))
    row = cur.fetchone()

    if row and row[0]:
        query_vector = np.array(row[0], dtype='float32')
        print(f"[LTL] Menggunakan refined vector dari database untuk hash: {img_hash}")
    else:
        query_vector = extract_query_vector(pil_image)
        cur.execute(
            "INSERT INTO image_history (image_hash, original_vector) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (img_hash, Json(query_vector.tolist()))
        )
        conn.commit()
        print(f"[LTL] Gambar baru terdeteksi, menyimpan original vector untuk hash: {img_hash}")
    
    cur.close()
    conn.close()

    results = similarity_search(query_vector, top_k=top_k)

    for r in results:
        r['image_b64'] = get_result_image_b64(r)
 
    return jsonify({
        'img_hash': img_hash,
        'query_image_b64': query_b64,
        'query_vector': query_vector.tolist(),
        'results': results
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'device': device, 'index_size': faiss_index.ntotal if faiss_index else 0})

@app.route('/refine', methods=['POST'])
def refine():
    data = request.get_json()
    img_hash = data.get('img_hash')
    original_vector = np.array(data['query_vector'], dtype='float32')
    relevant_indices = data['relevant_indices']
    irrelevant_indices = data['irrelevant_indices']
    top_k = data.get('top_k', 5)

    alpha, beta, gamma = 1.0, 0.8, 0.2

    refined = alpha * original_vector

    if relevant_indices:
        rel_vectors = np.array([
            faiss_index.reconstruct(int(i)) for i in relevant_indices
        ])
        refined += beta * rel_vectors.mean(axis=0)

    if irrelevant_indices:
        irrel_vectors = np.array([
            faiss_index.reconstruct(int(i)) for i in irrelevant_indices
        ])
        refined -= gamma * irrel_vectors.mean(axis=0)

    refined = l2_normalization(refined)

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            UPDATE image_history 
            SET refined_vector = %s,
                last_refined_at = NOW(),
                total_refinements = COALESCE(total_refinements, 0) + 1
            WHERE image_hash = %s
        """, (Json(refined.tolist()), img_hash))
        
        for idx in relevant_indices:
            cur.execute("""
                INSERT INTO users_feedback (query_hash, faiss_index, feedback_type, created_at)
                VALUES (%s, %s, 'relevant', NOW())
            """, (img_hash, int(idx)))

        for idx in irrelevant_indices:
            cur.execute("""
                INSERT INTO users_feedback (query_hash, faiss_index, feedback_type, created_at)
                VALUES (%s, %s, 'irrelevant', NOW())
            """, (img_hash, int(idx)))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"[DB] success updating refinement for hash: {img_hash}")
    except Exception as e:
        print(f"[DB Error] failed to save refinement: {e}")

    results = similarity_search(refined, top_k=top_k)
    for r in results:
        r['image_b64'] = get_result_image_b64(r)

    return jsonify({
        'results': results, 
        'refined_vector': refined.tolist()
    })

if __name__ == '__main__':
    load_assets()
    app.run(debug=False, host='0.0.0.0', port=5000)