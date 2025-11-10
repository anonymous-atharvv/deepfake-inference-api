from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np, io, tempfile, os, requests, cv2

# --- CONFIG ---
IMG_SIZE = 224

# ✅ Correct HuggingFace **DIRECT DOWNLOAD** link
MODEL_URL = "https://huggingface.co/anonymousananta/deepfake_model/resolve/main/model_epoch_03.h5"
MODEL_PATH = "model_epoch_03.h5"

# ✅ Download model once if not present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from HuggingFace...")
    r = requests.get(MODEL_URL)
    open(MODEL_PATH, "wb").write(r.content)
    print("✅ Model Downloaded Successfully!")

print("🔁 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model Loaded Successfully!")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

def preprocess_pil(pil_img, size=IMG_SIZE):
    pil_img = pil_img.convert("RGB").resize((size, size))
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(arr):
    p_real = float(model.predict(arr, verbose=0)[0][0])
    label = "REAL ✅" if p_real >= 0.5 else "FAKE ❌"
    confidence = p_real if p_real >= 0.5 else (1 - p_real)
    return {"label": label, "p_real": p_real, "confidence": confidence}

@app.get("/")
def root():
    return {"status": "API is working ✅"}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()
    pil = Image.open(io.BytesIO(data))
    arr = preprocess_pil(pil)
    return predict(arr)

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...), step: int = Query(15, ge=1, le=60)):
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name

    cap = cv2.VideoCapture(path)
    probs = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            arr = preprocess_pil(pil)
            probs.append(predict(arr)["p_real"])
        idx += 1

    cap.release()
    os.remove(path)

    if not probs:
        return {"error": "no_frames_detected"}

    mean_p = float(np.mean(probs))
    label = "REAL ✅" if mean_p >= 0.5 else "FAKE ❌"
    confidence = mean_p if mean_p >= 0.5 else (1 - mean_p)

    return {
        "label": label,
        "p_real_mean": mean_p,
        "confidence": confidence,
        "frames_sampled": len(probs),
        "step": step
    }
