#!/usr/bin/env python3
"""
read_truck_codes.py
Detecta camiones en una imagen, recorta cada uno, pregunta a moondream
el código/número que lleva pintado y dibuja el resultado sobre la imagen original.

Uso:
    python read_truck_codes.py <imagen> [--label truck] [--api http://localhost:8001] [--out resultado.jpg]
"""

import argparse
import io
import requests
import time
from PIL import Image, ImageDraw, ImageFont

API = "http://localhost:8001"
LABEL = "truck"
VQA_PROMPT = "What is the code or number written on this truck or trailer? Reply with only the code, nothing else. Ignore brand names (XPO Logistics, etc), license plates, capacity labels (cbm, kg) and irrelevant signs.If there is no code, reply with 'none'."


def detect(image_path: str, label: str, api: str) -> list:
    """Llama a /detect y devuelve lista de cajas YOLO."""
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{api}/detect",
            files={"image": f},
            data={"label": label},
            timeout=120,
        )
    r.raise_for_status()
    return r.json()["detections"]  # [[label, xc, yc, w, h], ...]


def vqa_crop(crop: Image.Image, prompt: str, api: str) -> str:
    """Envía un crop PIL a /vqa y devuelve la respuesta."""
    buf = io.BytesIO()
    crop.save(buf, format="JPEG")
    buf.seek(0)
    r = requests.post(
        f"{api}/vqa",
        files={"image": ("crop.jpg", buf, "image/jpeg")},
        data={"prompt": prompt},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["answer"].strip()


def yolo_to_pixels(xc, yc, w, h, img_w, img_h):
    """Convierte caja YOLO normalizada a píxeles (x1, y1, x2, y2)."""
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    # Clamp a los bordes
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Ruta a la imagen de entrada")
    parser.add_argument("--label", default=LABEL, help="Etiqueta a detectar (default: truck)")
    parser.add_argument("--api", default=API, help="URL base de la API")
    parser.add_argument("--out", default="resultado.jpg", help="Imagen de salida")
    parser.add_argument("--prompt", default=VQA_PROMPT, help="Prompt VQA personalizado")
    args = parser.parse_args()

    start_total = time.time()

    img = Image.open(args.image).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    # Intentar cargar fuente grande, si no hay usar la default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # Medir tiempo de detección
    print(f"[INFO] Detectando '{args.label}'...")
    start_det = time.time()
    detections = detect(args.image, args.label, args.api)
    end_det = time.time()
    
    print(f"[INFO] {len(detections)} detecciones en {end_det - start_det:.2f} segundos")

    vqa_times = []

    for i, det in enumerate(detections):
        _, xc, yc, w, h = det
        x1, y1, x2, y2 = yolo_to_pixels(xc, yc, w, h, img_w, img_h)

        if x2 <= x1 or y2 <= y1:
            print(f"  [{i+1}] Caja inválida, saltando")
            continue

        crop = img.crop((x1, y1, x2, y2))

        start_vqa = time.time()

        print(f"  [{i+1}] Crop ({x1},{y1})-({x2},{y2}) → VQA ...")
        code = vqa_crop(crop, args.prompt, args.api)
        print(f"  [{i+1}] Código: {code}")

        end_vqa = time.time()

        duration_vqa = end_vqa - start_vqa
        vqa_times.append(duration_vqa)
        print(f"  [{i+1}] Código: {code} (Tardó: {duration_vqa:.2f}s)")

        # Dibujar caja
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

        # Dibujar etiqueta con fondo negro
        label_text = code if code.lower() != "none" else "?"
        bbox = draw.textbbox((x1, y1 - 32), label_text, font=font)
        draw.rectangle(bbox, fill="black")
        draw.text((x1, y1 - 32), label_text, fill="lime", font=font)

    img.save(args.out, format="JPEG", quality=92)
    print(f"\n[OK] Resultado guardado en: {args.out}")
    # Finalizar cronómetro total
    end_total = time.time()
    total_duration = end_total - start_total
    print("\n" + "="*30)
    print(f"RESUMEN DE TIEMPOS:")
    print(f"⏱️ Detección inicial: {end_det - start_det:.2f}s")
    if vqa_times:
        print(f"⏱️ VQA promedio por camión: {sum(vqa_times)/len(vqa_times):.2f}s")
    print(f"🚀 TIEMPO TOTAL: {total_duration:.2f}s")
    print("="*30)


if __name__ == "__main__":
    main()