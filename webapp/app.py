"""
Interfaz web local para el Auditor de ROI de Patrocinadores.

Permite subir un vídeo desde el navegador y corre el pipeline completo
(detección -> analítica -> informe) sin tocar la terminal.

Uso:
    cd webapp
    python app.py
    # abre http://127.0.0.1:5000 en el navegador

Nota: pensado para uso local de un solo desarrollador (sin autenticación,
sin cola de tareas). El procesamiento es síncrono: la página se queda
"cargando" mientras el vídeo se analiza (normal, puede tardar 1-2 minutos
para un clip corto en CPU).
"""

import sys
import uuid
from pathlib import Path

from flask import Flask, request, redirect, url_for, send_file, abort
from werkzeug.utils import secure_filename
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import RAW_CLIPS_DIR, REPORTS_DIR, OUTPUT_DIR  # noqa: E402
import detect_logos  # noqa: E402
import analytics  # noqa: E402
import report as report_module  # noqa: E402

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB, de sobra para un clip corto

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

PAGE_STYLE = """
<style>
  .viz-root {
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b;
    --text-secondary: #52514e; --muted: #898781; --grid: #e1e0d9;
    --baseline: #c3c2b7; --border: rgba(11,11,11,0.10); --accent: #2a78d6;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark; --surface-1: #1a1a19; --page: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
      --accent: #3987e5;
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark; --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --accent: #3987e5;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .viz-root { max-width: 640px; margin: 0 auto; padding: 60px 24px; color: var(--text-primary); }
  .eyebrow { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 6px; }
  h1 { font-size: 26px; margin: 0 0 10px; font-weight: 650; }
  p.lead { color: var(--text-secondary); font-size: 15px; line-height: 1.5; margin: 0 0 32px; }
  .dropzone {
    background: var(--surface-1); border: 1.5px dashed var(--baseline); border-radius: 12px;
    padding: 40px 24px; text-align: center; transition: border-color .15s;
  }
  .dropzone.dragover { border-color: var(--accent); }
  .dropzone input[type=file] { display: none; }
  .dropzone label { cursor: pointer; color: var(--accent); font-weight: 600; }
  #filename { margin-top: 12px; font-size: 13.5px; color: var(--text-secondary); }
  button {
    margin-top: 20px; width: 100%; padding: 12px 20px; border: none; border-radius: 8px;
    background: var(--accent); color: white; font-size: 15px; font-weight: 600; cursor: pointer;
    font-family: inherit;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .field { margin-top: 16px; }
  .field label { display: block; font-size: 13px; color: var(--text-secondary); margin-bottom: 6px; }
  .field input[type=text] {
    width: 100%; padding: 9px 12px; border: 1px solid var(--border); border-radius: 6px;
    font-size: 14px; background: var(--surface-1); color: var(--text-primary); font-family: inherit;
  }
  .status { display:none; margin-top: 20px; text-align: center; color: var(--text-secondary); font-size: 14px; }
  .status.active { display: block; }
  .error { background: #fdeceb; border: 1px solid #e34948; color: #8a2323; padding: 12px 16px;
           border-radius: 8px; font-size: 13.5px; margin-bottom: 20px; }
  a.back { color: var(--muted); font-size: 13px; text-decoration: none; }
  footer { margin-top: 40px; color: var(--muted); font-size: 12px; }
</style>
"""

UPLOAD_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auditor de ROI de Patrocinadores</title>
{style}
<div class="viz-root">
  <p class="eyebrow">Auditor de ROI de patrocinadores &middot; visión artificial</p>
  <h1>Sube un clip para analizarlo</h1>
  <p class="lead">Detecta logos de patrocinador, mide segundos de exposición legible y genera el
  informe automáticamente. Formatos: MP4, MOV, AVI, MKV, WEBM.</p>

  {error_html}

  <form action="/analizar" method="post" enctype="multipart/form-data" id="uploadForm">
    <div class="dropzone" id="dropzone">
      <label for="videoInput">Elige un vídeo o arrástralo aquí</label>
      <input type="file" name="video" id="videoInput" accept="video/*" required>
      <div id="filename">Ningún archivo seleccionado</div>
    </div>
    <div class="field">
      <label for="clipName">Nombre del clip (opcional, para el informe)</label>
      <input type="text" name="clip_name" id="clipName" placeholder="Ej. Onboard - Curva 4">
    </div>
    <button type="submit" id="submitBtn">Analizar vídeo</button>
    <div class="status" id="status">Procesando vídeo... esto puede tardar 1-2 minutos, no cierres esta pestaña.</div>
  </form>

  <footer>El vídeo se procesa localmente en tu máquina, no se sube a ningún servidor externo.</footer>
</div>
<script>
  const input = document.getElementById('videoInput');
  const filenameEl = document.getElementById('filename');
  const dropzone = document.getElementById('dropzone');
  const form = document.getElementById('uploadForm');
  const submitBtn = document.getElementById('submitBtn');
  const status = document.getElementById('status');

  input.addEventListener('change', () => {{
    filenameEl.textContent = input.files.length ? input.files[0].name : 'Ningún archivo seleccionado';
  }});
  ['dragover'].forEach(evt => dropzone.addEventListener(evt, e => {{
    e.preventDefault(); dropzone.classList.add('dragover');
  }}));
  ['dragleave', 'drop'].forEach(evt => dropzone.addEventListener(evt, e => {{
    e.preventDefault(); dropzone.classList.remove('dragover');
  }}));
  dropzone.addEventListener('drop', e => {{
    if (e.dataTransfer.files.length) {{
      input.files = e.dataTransfer.files;
      filenameEl.textContent = input.files[0].name;
    }}
  }});
  form.addEventListener('submit', () => {{
    submitBtn.disabled = true;
    submitBtn.textContent = 'Procesando...';
    status.classList.add('active');
  }});
</script>
"""


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return UPLOAD_PAGE.format(style=PAGE_STYLE, error_html="")


@app.route("/analizar", methods=["POST"])
def analizar():
    file = request.files.get("video")
    clip_name = (request.form.get("clip_name") or "Clip analizado").strip() or "Clip analizado"

    if file is None or file.filename == "":
        return UPLOAD_PAGE.format(
            style=PAGE_STYLE,
            error_html='<div class="error">No se seleccionó ningún archivo de vídeo.</div>',
        ), 400

    if not allowed_file(file.filename):
        return UPLOAD_PAGE.format(
            style=PAGE_STYLE,
            error_html='<div class="error">Formato no soportado. Usa MP4, MOV, AVI, MKV o WEBM.</div>',
        ), 400

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    video_path = RAW_CLIPS_DIR / unique_name
    file.save(video_path)

    # Duración real del vídeo (nada de pedírsela al usuario a mano)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps else 0
    cap.release()

    if duration <= 0:
        return UPLOAD_PAGE.format(
            style=PAGE_STYLE,
            error_html='<div class="error">No se pudo leer el vídeo. Prueba con otro archivo o formato.</div>',
        ), 400

    try:
        detect_logos.run(video_path, save_sample_frames=True)
        analytics.run(OUTPUT_DIR / "detections.csv", duration)
        report_module.run(duration, clip_name)
    except Exception as exc:  # noqa: BLE001
        return UPLOAD_PAGE.format(
            style=PAGE_STYLE,
            error_html=f'<div class="error">Error procesando el vídeo: {exc}</div>',
        ), 500

    return redirect(url_for("ver_informe"))


@app.route("/informe")
def ver_informe():
    report_path = REPORTS_DIR / "sponsor_roi_report.html"
    if not report_path.exists():
        return redirect(url_for("index"))
    return send_file(report_path)


if __name__ == "__main__":
    print("Abre http://127.0.0.1:5000 en tu navegador")
    app.run(host="127.0.0.1", port=5000, debug=False)
