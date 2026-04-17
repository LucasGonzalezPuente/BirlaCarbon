# moondream2-api
API ligera en FastAPI para describir imágenes (caption), detectar objetos por etiqueta, localizar puntos y hacer VQA (preguntas sobre la imagen)

Permite elegir modelo base o modelo cuantizado 4-bit, y además incluye endpoints que pintan resultados directamente sobre la imagen (cajas y puntos).

⚠️ Warmup / primer uso: la primera petición tras arrancar (o tras cambiar de modelo) tarda sensiblemente más: se descargan pesos, se inicializan kernels y se calienta el grafo. A partir de la segunda llamada, la latencia baja notablemente.

## Características ✨
Dos modos de modelo:

* `vikhyatk/moondream2 (completo, mayor compatibilidad).
* `moondream/moondream-2b-2025-04-14-4bit (cuantizado 4-bit, más ligero).

Salidas en formato YOLO: `[label, x_center, y_center, width, height]` normalizado en `[0,1]`.

Endpoints de imagen que devuelven un JPEG con resultados dibujados (cajas/puntos + etiquetas).

Swagger/OpenAPI en `/docs` y `/openapi.json`.

Health-check con información de dispositivo, modelo cargado y cuantización.

## Requisitos 📦
* Python + dependencias (véase `requirements.txt`): `fastapi`, `uvicorn[standard]`, `transformers>=4.41.0`, `accelerate>=0.30.0`, `safetensors`, `torchao`, `bitsandbytes>=0.43.0`, `pillow`, `numpy`, `huggingface_hub>=0.23`, `python-multipart`.
* Opcional GPU con drivers NVIDIA (Docker Compose ya reserva un GPU y expone capacidades apropiadas).

## Puesta en marcha 🚀
Opción A) Docker / Docker Compose (recomendado)

1. Estructura mínima
```bash
.
├─ main.py
├─ requirements.txt
├─ docker-compose.yaml
└─ Dockerfile
```

2. Arranca el servicio
```bash
docker compose up --build -d
```

El servicio expone `http://localhost:8001` (mapea `8001:8000`).

Cache de modelos en `./models` montado en `/models` dentro del contenedor (HF cache en `/models/hf`).

Variables útiles:

`HF_HOME` y HUGGINGFACE_HUB_CACHE → `/models/hf` (persistencia entre reinicios).

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (gestión de memoria CUDA).

3. Probar

Swagger: `http://localhost:8001/docs`

Health:
```bash
curl -s http://localhost:8001/health | jq
```

Respuesta típica:
```bash
{
  "ok": true,
  "device": "cuda",
  "gpu_name": "NVIDIA ...",
  "model_id": "vikhyatk/moondream2",
  "quantization": "fp32",
  "version": "3.5.0"
}
```

Opción B) Local (sin Docker)

1. Instala dependencias:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Arranca Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. Abre `http://localhost:8000/docs`.

## Selección y descarga de modelos 🔁 

* Descargar pesos (cache local Hugging Face):
```bash
curl -X POST http://localhost:8001/download_model \
  -F model=full   # o 'quant'
```

Devuelve ruta del cache.

* Cambiar modelo activo (sin reiniciar):
```bash
curl -X POST http://localhost:8001/set_model \
  -F model=quant  # o 'full'
```

El health reflejará model_id y quantization actualizados.

Nota: El cambio de modelo provoca warmup en la siguiente petición.

## Endpoints 🧠
`GET /health`

Devuelve estado del modelo y del dispositivo (`cpu/cuda`), nombre de GPU, `model_id`, `quantization` y `version`.

`POST /caption`

Genera una descripción de la imagen.

Campos (multipart/form-data):

* `image` (file) — obligatorio

* `length` (enum: `short|normal|long`, por defecto `normal`)

Ejemplo:
```bash
curl -X POST http://localhost:8001/caption \
  -F image=@examples/cat.jpg \
  -F length=short
```

Respuesta:
```bash
{"caption": "...", "length": "short"}
```

POST /detect
```bash
Detecta objetos de una etiqueta concreta y devuelve cajas en formato YOLO.
```
Campos:

image (file) — obligatorio

label (string) — obligatorio (p. ej. "person", "car")

Respuesta:
```bash
{
  "detections": [
    ["person", x_center, y_center, width, height],
    ...
  ],
  "n_boxes": 2
}
```

Valores normalizados en [0,1] respecto a ancho/alto.

Ejemplo:
```bash
curl -X POST http://localhost:8001/detect \
  -F image=@examples/street.jpg \
  -F label=person
```
POST /point

Localiza puntos para una etiqueta (p. ej., “ball”, “logo”) y devuelve entradas tipo YOLO con ancho/alto 0.0 (punto).

Campos: image (file), label (string)

Respuesta:
```bash
{
  "detections": [
    ["logo", x_center, y_center, 0.0, 0.0], ...
  ],
  "n_points": 1
}
```

POST /detect_image

Como /detect, pero devuelve un JPEG con cajas y etiquetas dibujadas.
Ideal para verificación visual rápida.

Campos: image (file), label (string)

Ejemplo:
```bash
curl -X POST http://localhost:8001/detect_image \
  -F image=@examples/street.jpg \
  -F label=car \
  -o out_detect.jpg
```

Salida: out_detect.jpg con cajas verdes y etiquetas en la esquina.

POST /point_image

Como /point, pero devuelve un JPEG con puntos y etiquetas dibujados.

Ejemplo:
```bash
curl -X POST http://localhost:8001/point_image \
  -F image=@examples/logo.png \
  -F label=logo \
  -o out_points.jpg
```

Dibuja círculos rojos y texto de la etiqueta.

POST /vqa

Responde a una pregunta libre sobre la imagen (Visual Question Answering).

Campos:

image (file)

prompt (string) — obligatorio

Ejemplo:
```bash
curl -X POST http://localhost:8001/vqa \
  -F image=@examples/kitchen.jpg \
  -F prompt="How many cups are on the table?"
```

Respuesta:

{"answer": "Two", "prompt": "How many cups are on the table?"}


## Formato YOLO (recordatorio) 📐

Cada detección es:
[label, x_center, y_center, width, height]
donde todos los valores están normalizados a [0,1] con respecto al tamaño de la imagen. Los endpoints de “imagen” convierten de normalizado a píxeles para dibujar correctamente.

## Configuración y caché de modelos 🔧 

* El contenedor monta ./models en /models, y define HF_HOME y HUGGINGFACE_HUB_CACHE a /models/hf para reutilizar descargas entre reinicios.

* Si quieres forzar CPU, puedes arrancar con CUDA_VISIBLE_DEVICES="" en el servicio. (Basta ajustar la variable en docker-compose.yaml).

## Troubleshooting 🐛 

* Primera llamada muy lenta → esperado por warmup, pesos y compilación. Consulta /health para verificar el estado antes de lanzar cargas.

* Cuantizado 4-bit en GPUs antiguas (SM 6.x, p. ej. Pascal): si detectas errores CUDA al usar el modelo cuantizado, cambia al modelo base (full) o fuerza CPU. También puedes explorar carga en 8-bit con bitsandbytes si ajustas el bloque de carga de modelos.

* CORS está abierto a cualquier origen por defecto. Ajusta la lista en main.py si lo expones públicamente.

## Rutas rápidas (chuleta) 🧭

* GET /health — estado y metadatos.

* POST /download_model — descarga y cachea pesos (model=full|quant).

* POST /set_model — selecciona el modelo activo (model=full|quant).

* POST /caption — descripción (length=short|normal|long).

* POST /detect — cajas YOLO para label.

* POST /point — puntos YOLO para label.

* POST /detect_image — JPEG con cajas + etiquetas.

* POST /point_image — JPEG con puntos + etiquetas.

## Detalles técnicos 🏗️ 

* Servidor: FastAPI + Uvicorn. CORS permisivo por defecto.

* Carga del modelo: AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True) con device_map automático en CUDA y reintentos con fallback a CPU.

* Versión de la app: 3.5.0 (expuesta en /health).

* Dependencias: ver requirements.txt (incluye torchao y bitsandbytes para cuantización).

* Docker Compose: expone 8001:8000, reserva 1 GPU, define caché HF y configura memoria CUDA
