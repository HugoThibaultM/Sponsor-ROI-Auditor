"""
Pipeline de detección + tracking de logos de patrocinador en vídeo.

Uso:
    python src/detect_logos.py --video data/raw_clips/mi_clip.mp4

Salida:
    output/detections.csv       -> una fila por detección (frame, tiempo, clase, bbox, conf, nitidez)
    output/sample_frames/*.jpg  -> frames anotados de muestra (para verificación visual)

Enfoque técnico (ver STRATEGY.md para el detalle completo):
  - Detección zero-shot con YOLO-World (open-vocabulary): permite detectar
    "Oracle logo", "Petronas logo", etc. por texto, sin dataset propio.
    Es el atajo correcto para un prototipo de una persona en pocos días.
    Para precisión de nivel producción (lluvia, chispas, vibración de
    cámara a 300km/h) hace falta fine-tuning con dataset propio etiquetado.
  - Tracking con ByteTrack (incluido en Ultralytics) para no contar el
    mismo logo dos veces frame a frame, y para poder medir "rachas"
    continuas de visibilidad (ej. "42 segundos ininterrumpidos").
  - Métrica de nitidez (varianza del Laplaciano) por bbox, como proxy de
    si el logo es realmente "legible" y no solo detectado bajo motion blur.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SPONSOR_PROMPTS,
    YOLO_WORLD_MODEL,
    SAMPLE_FPS,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    OUTPUT_DIR,
)


def sharpness_score(gray_crop: np.ndarray) -> float:
    """Varianza del Laplaciano: valores bajos ~ imagen borrosa (motion blur)."""
    if gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def build_class_list():
    """Aplana SPONSOR_PROMPTS -> lista de prompts + mapa prompt->marca."""
    prompts = []
    prompt_to_brand = {}
    for brand, brand_prompts in SPONSOR_PROMPTS.items():
        for p in brand_prompts:
            prompts.append(p)
            prompt_to_brand[p] = brand
    return prompts, prompt_to_brand


def run(video_path: Path, save_sample_frames: bool = True, max_sample_frames: int = 12):
    prompts, prompt_to_brand = build_class_list()

    print(f"[1/4] Cargando modelo zero-shot {YOLO_WORLD_MODEL} con {len(prompts)} prompts...")
    model = YOLO(YOLO_WORLD_MODEL)
    # YOLOE necesita los embeddings de texto explícitos (get_text_pe) en vez
    # de solo la lista de nombres, a diferencia de la API antigua de YOLO-World.
    if hasattr(model, "get_text_pe"):
        model.set_classes(prompts, model.get_text_pe(prompts))
    else:
        model.set_classes(prompts)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el vídeo: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = max(width * height, 1)

    frame_step = max(int(round(native_fps / SAMPLE_FPS)), 1)
    print(f"[2/4] Vídeo: {width}x{height} @ {native_fps:.1f}fps, {total_frames} frames.")
    print(f"      Procesando 1 de cada {frame_step} frames (~{SAMPLE_FPS} fps efectivos).")

    detections_path = OUTPUT_DIR / "detections.csv"
    sample_dir = OUTPUT_DIR / "sample_frames"
    sample_dir.mkdir(exist_ok=True)

    rows = []
    frame_idx = 0
    processed = 0
    sample_saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        t_seconds = frame_idx / native_fps

        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        r = results[0]

        annotated = frame.copy()
        n_dets_this_frame = 0

        if r.boxes is not None and len(r.boxes) > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                brand = prompt_to_brand.get(cls_name, cls_name)
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, width), min(y2, height)
                crop = gray[y1:y2, x1:x2]
                sharpness = sharpness_score(crop)
                area_fraction = ((x2 - x1) * (y2 - y1)) / frame_area

                rows.append({
                    "frame_idx": frame_idx,
                    "t_seconds": round(t_seconds, 3),
                    "brand": brand,
                    "prompt_class": cls_name,
                    "track_id": track_id,
                    "conf": round(conf, 3),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area_fraction": round(area_fraction, 5),
                    "sharpness": round(sharpness, 1),
                })
                n_dets_this_frame += 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                label = f"{brand} #{track_id} {conf:.2f}"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

        if save_sample_frames and n_dets_this_frame > 0 and sample_saved < max_sample_frames:
            out_path = sample_dir / f"frame_{frame_idx:06d}_t{t_seconds:.2f}s.jpg"
            cv2.imwrite(str(out_path), annotated)
            sample_saved += 1

        processed += 1
        if processed % 25 == 0:
            print(f"      ...procesados {processed} frames muestreados "
                  f"(t={t_seconds:.1f}s / {total_frames/native_fps:.1f}s)")

        frame_idx += 1

    cap.release()

    print(f"[3/4] Detecciones totales: {len(rows)}. Guardando CSV...")
    with open(detections_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_idx", "t_seconds", "brand", "prompt_class", "track_id",
            "conf", "x1", "y1", "x2", "y2", "area_fraction", "sharpness",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[4/4] Listo. CSV: {detections_path}")
    print(f"      Frames de muestra anotados: {sample_dir} ({sample_saved} guardados)")
    return detections_path, native_fps, (width, height), total_frames / native_fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detección de logos de patrocinador en vídeo (zero-shot).")
    parser.add_argument("--video", required=True, help="Ruta al clip de vídeo a analizar.")
    parser.add_argument("--no-samples", action="store_true", help="No guardar frames de muestra anotados.")
    args = parser.parse_args()

    run(Path(args.video), save_sample_frames=not args.no_samples)
