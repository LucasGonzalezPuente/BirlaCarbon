import io
import os
import time
import shutil
import threading
from enum import Enum
from typing import Any, Dict, Optional, Tuple, List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

# ============================================================
# Configuración
# ============================================================
DEFAULT_MODEL = "moondream/moondream-2b-2025-04-14-4bit"
MODEL_ID = DEFAULT_MODEL
model: Optional[AutoModelForCausalLM] = None
DEVICE: str = "cpu"
GPU_NAME: Optional[str] = None
APP_VERSION = "3.5.0"
QUANTIZATION = "4bit-qat" if "4bit" in MODEL_ID else "fp32"

# Serializa el acceso a la GPU: solo una inferencia a la vez.
# FastAPI ejecuta handlers sync en un threadpool, y sin este lock
# las peticiones concurrentes pueden solapar forward passes en la
# misma GPU → OOM, resultados corruptos o crashes.
_inference_lock = threading.Lock()

# ============================================================
# ModelChoice para Swagger
# ============================================================
class ModelChoice(str, Enum):
    full = "vikhyatk/moondream2"
    quant = "moondream/moondream-2b-2025-04-14-4bit"

# ============================================================
# Utilidades
# ============================================================
def _hf_home() -> str:
    return os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

def _clear_hf_dynamic_cache(model_id: str):
    home = _hf_home()
    try:
        org, repo = model_id.split("/", 1)
    except ValueError:
        return
    dyn_dir = os.path.join(home, "modules", "transformers_modules", org, repo)
    snap_dir = os.path.join(home, "hub", f"models--{org}--{repo}")
    for path in (dyn_dir, snap_dir):
        try:
            shutil.rmtree(path)
            print(f"[INFO] Cleared HF cache: {path}")
        except FileNotFoundError:
            pass

def _probe_cuda(retries: int = 6, pause: float = 1.0) -> Tuple[bool, Optional[str]]:
    for i in range(retries):
        try:
            if torch.cuda.is_available():
                _ = torch.randn(1, device="cuda")
                return True, torch.cuda.get_device_name(0)
        except Exception as e:
            print(f"[WARN] CUDA probe failed ({i+1}/{retries}): {e}")
        time.sleep(pause)
    return False, None

def _maybe_compile(m: AutoModelForCausalLM):
    if DEVICE != "cuda":
        return
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return
    if major >= 7:
        print("[INFO] compile() disabled manually; running without torch.compile")
    else:
        print("[INFO] compile disabled on SM<70")

def _load_model(model_id: str):
    global model, DEVICE, GPU_NAME, MODEL_ID, QUANTIZATION
    has_cuda, GPU_NAME = _probe_cuda()
    DEVICE = "cuda" if has_cuda else "cpu"
    MODEL_ID = model_id
    QUANTIZATION = "4bit-qat" if "4bit" in MODEL_ID else "fp32"
    print(f"[INFO] Loading {MODEL_ID} on {DEVICE}")

    attempts = 0
    while attempts < 2:
        attempts += 1
        try:
            m = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision="main",
                trust_remote_code=True,
                device_map={"": "cuda"} if DEVICE == "cuda" else None,
                low_cpu_mem_usage=True,
                force_download=(attempts == 2),
                local_files_only=False,
            )
            m.eval()
            model = m
            print(f"[INFO] Model loaded on {DEVICE} ({GPU_NAME or 'CPU'}) [{QUANTIZATION}]")
            _maybe_compile(model)
            return
        except FileNotFoundError as e:
            print(f"[WARN] Missing dynamic module file: {e}")
            _clear_hf_dynamic_cache(MODEL_ID)
        except Exception as e:
            print(f"[WARN] Load attempt {attempts} failed on {DEVICE}: {e}")
            if DEVICE == "cuda":
                print("[INFO] Falling back to CPU and retrying...")
                DEVICE, GPU_NAME = "cpu", None
    raise RuntimeError("Failed to load model after retries.")

def _read_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

def _ensure_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

def _save_temp_image(pil_img: Image.Image, prefix: str) -> str:
    out_path = f"/tmp/{prefix}_{int(time.time()*1000)}.jpg"
    pil_img.save(out_path, format="JPEG")
    return out_path

# ============================================================
# Conversión a formato YOLO
# ============================================================
def to_yolo_box(x_min: float, y_min: float, x_max: float, y_max: float, label: str) -> List[float]:
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    return [label, x_center, y_center, width, height]

def point_to_yolo(pt: Dict[str, float], label: str) -> List[float]:
    return [label, pt["x"], pt["y"], 0.0, 0.0]

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(
    title="Moondream2 API (YOLO format)",
    description="API con Moondream2 normal/cuanti. Endpoints devuelven en formato YOLO. Selección de modelo desde Swagger.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    try:
        _load_model(MODEL_ID)
    except Exception as e:
        print(f"[ERROR] Application startup failed to load model: {e}")

# ============================================================
# Endpoints
# ============================================================
class CaptionLength(str, Enum):
    short = "short"
    normal = "normal"
    long = "long"

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": model is not None,
        "device": DEVICE,
        "gpu_name": GPU_NAME,
        "model_id": MODEL_ID,
        "quantization": QUANTIZATION,
        "version": APP_VERSION,
    }

@app.post("/download_model")
def download_model(model: ModelChoice = Form(...)) -> Dict[str, Any]:
    path = snapshot_download(repo_id=model.value)
    return {"ok": True, "model": model.value, "cached_at": path}

@app.post("/set_model")
def set_model(model: ModelChoice = Form(...)) -> Dict[str, Any]:
    with _inference_lock:
        _load_model(model.value)
    return {"ok": True, "model_id": MODEL_ID, "quantization": QUANTIZATION}

@app.post("/caption")
def caption(
    image: UploadFile = File(...),
    length: CaptionLength = Form(CaptionLength.normal),
) -> Dict[str, Any]:
    _ensure_model()
    img = _read_image(image)
    with _inference_lock, torch.inference_mode():
        out = model.caption(img, length=length.value, stream=False)  # type: ignore[attr-defined]
    text = out["caption"] if isinstance(out, dict) and "caption" in out else out
    return {"caption": text, "length": length.value}

@app.post("/detect")
def detect(
    image: UploadFile = File(...),
    label: str = Form(...),
) -> Dict[str, Any]:
    _ensure_model()
    img = _read_image(image)
    with _inference_lock, torch.inference_mode():
        res = model.detect(img, label)  # type: ignore[attr-defined]
    objs: List[Dict[str, float]] = res.get("objects", []) if isinstance(res, dict) else []
    yolo_boxes = [to_yolo_box(o["x_min"], o["y_min"], o["x_max"], o["y_max"], label) for o in objs]
    return {"detections": yolo_boxes, "n_boxes": len(yolo_boxes)}

@app.post("/point")
def point(
    image: UploadFile = File(...),
    label: str = Form(...),
) -> Dict[str, Any]:
    _ensure_model()
    img = _read_image(image)
    with _inference_lock, torch.inference_mode():
        res = model.point(img, label)  # type: ignore[attr-defined]
    pts: List[Dict[str, float]] = res.get("points", []) if isinstance(res, dict) else []
    yolo_points = [point_to_yolo(pt, label) for pt in pts]
    return {"detections": yolo_points, "n_points": len(pts)}

@app.post("/point_image")
def point_image(
    image: UploadFile = File(...),
    label: str = Form(...),
):
    _ensure_model()
    img = _read_image(image)
    with _inference_lock, torch.inference_mode():
        res = model.point(img, label)  # type: ignore[attr-defined]
    pts: List[Dict[str, float]] = res.get("points", []) if isinstance(res, dict) else []
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for pt in pts:
        x, y = pt["x"] * w, pt["y"] * h
        r = 6
        draw.ellipse((x-r, y-r, x+r, y+r), fill="red", outline="white")
        draw.text((x+r, y), label, fill="yellow")
    out_path = _save_temp_image(img, "point")
    return FileResponse(out_path, media_type="image/jpeg")

@app.post("/detect_image")
def detect_image(
    image: UploadFile = File(...),
    label: str = Form(...),
):
    _ensure_model()
    img = _read_image(image)
    with _inference_lock, torch.inference_mode():
        res = model.detect(img, label)  # type: ignore[attr-defined]
    objs: List[Dict[str, float]] = res.get("objects", []) if isinstance(res, dict) else []
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for o in objs:
        x1, y1 = o["x_min"] * w, o["y_min"] * h
        x2, y2 = o["x_max"] * w, o["y_max"] * h
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1, y1), label, fill="yellow")
    out_path = _save_temp_image(img, "detect")
    return FileResponse(out_path, media_type="image/jpeg")

@app.post("/vqa")
def vqa(
    image: UploadFile = File(...),
    prompt: str = Form(...),
) -> Dict[str, Any]:
    _ensure_model()

    # -----------------------------
    # LOG 1: inicio de la petición
    # -----------------------------
    t0 = time.time()
    print("\n==========================")
    print("[VQA] Request received")
    print("==========================")
    print(f"[VQA] Prompt: {prompt}")

    # -----------------------------
    # LOG 2: leer la imagen
    # -----------------------------
    t_read0 = time.time()
    img = _read_image(image)
    t_read1 = time.time()
    print(f"[VQA] Image read in {t_read1 - t_read0:.3f} sec")

    # -----------------------------
    # LOG 3: inferencia
    # -----------------------------
    print("[VQA] Starting model.query() ...")
    t_inf0 = time.time()

    with _inference_lock, torch.inference_mode():
        ans = model.query(img, prompt)  # type: ignore[attr-defined]

    t_inf1 = time.time()
    print(f"[VQA] model.query() finished in {t_inf1 - t_inf0:.3f} sec")

    # -----------------------------
    # LOG 4: preparar respuesta
    # -----------------------------
    text = ans["answer"] if isinstance(ans, dict) and "answer" in ans else ans

    # -----------------------------
    # LOG 5: fin de petición
    # -----------------------------
    t1 = time.time()
    print(f"[VQA] Answer: {text}")
    print(f"[VQA] TOTAL request time: {t1 - t0:.3f} sec")
    print("===========================================\n")

    return {"answer": text, "prompt": prompt, "latency_sec": round(t1 - t0, 4)}
