"""
Configuración del Auditor de ROI para Patrocinadores (Visión Artificial).

Ajusta SPONSOR_PROMPTS con los nombres de marca / descripciones textuales
que quieras rastrear. El modelo usado (YOLO-World) es "open-vocabulary":
no hace falta re-entrenarlo para probar con marcas nuevas, basta con
cambiar estos prompts. Para producción real (precisión alta en lluvia,
chispas, 300km/h, motion blur) hay que sustituir esto por un modelo
YOLOv8/v9 fine-tuneado con un dataset propio etiquetado — ver
STRATEGY.md para el plan de datos y coste en la nube.
"""

from pathlib import Path

# --- Rutas del proyecto ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CLIPS_DIR = DATA_DIR / "raw_clips"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

for d in [DATA_DIR, RAW_CLIPS_DIR, OUTPUT_DIR, REPORTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Modelo de detección ---
# Modelo zero-shot (open-vocabulary): YOLOE con backend de texto MobileCLIP.
# Se usa YOLOE en vez de YOLO-World porque YOLO-World depende de los pesos
# oficiales de OpenAI CLIP (host bloqueado en algunos entornos/redes
# corporativas); YOLOE + MobileCLIP se descarga desde GitHub Releases de
# Ultralytics, con mejor compatibilidad de red. Cambia a un .pt propio
# (fine-tuned) cuando tengas un dataset etiquetado real de logos de F1.
YOLO_WORLD_MODEL = "yoloe-11s-seg.pt"

# --- Patrocinadores a rastrear (edítalo por equipo/carrera) ---
# Cada entrada: nombre a mostrar en el reporte -> lista de prompts de texto
# (varias formas de describir el mismo logo mejoran el recall zero-shot).
SPONSOR_PROMPTS = {
    "Oracle": ["Oracle logo", "Oracle red text logo"],
    "Red Bull": ["Red Bull logo", "Red Bull can logo", "Red Bull bulls logo"],
    "Petronas": ["Petronas logo", "Petronas green logo"],
    "Sponsor genérico (fallback)": ["sponsor logo on race car", "brand logo on car livery"],
}

# --- Parámetros de procesamiento de vídeo ---
# Se procesa a un FPS reducido para que sea viable en CPU. Para el reporte
# final de producción se recomienda GPU + FPS nativo del broadcast (25/50).
SAMPLE_FPS = 5
CONF_THRESHOLD = 0.12          # YOLO-World zero-shot suele necesitar umbrales bajos
IOU_THRESHOLD = 0.5

# Tolerancia de "hueco" entre detecciones del mismo track para seguir
# contando como una racha continua (cubre parpadeos, motion blur, chispas).
MAX_GAP_SECONDS = 0.6

# Legibilidad: qué % mínimo del frame debe ocupar la bbox del logo para
# considerarlo "legible" en un informe de TV (proxy simple, ajustable).
MIN_AREA_FRACTION_LEGIBLE = 0.0015  # 0.15% del área del frame

# Valor mediático equivalente ilustrativo (USD por segundo de exposición
# legible en TV mundial). Cifra de ejemplo para el reporte demo — en un
# informe real se sustituye por datos de la agencia de medición de audiencia
# del equipo (ej. Nielsen Sponsorship, Hookit, GumGum Sports).
ILLUSTRATIVE_USD_PER_SECOND = 1450
