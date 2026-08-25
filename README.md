# Auditor de ROI para Patrocinadores (Visión Artificial)

Prototipo funcional: detecta logos de patrocinador en un clip de vídeo,
calcula segundos de exposición legible por marca (incluida la "racha
ininterrumpida más larga"), y genera un informe listo para enseñar al equipo
comercial de una escudería.

Detección zero-shot con **YOLOE + MobileCLIP** (por texto, sin dataset propio)
+ tracking con **ByteTrack**. Ver `STRATEGY.md` para el plan completo de paso
a producción (dataset propio, fine-tuning, coste en la nube) y la estrategia
de marketing en solitario para conseguir la primera reunión.

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate          # Windows (PowerShell): venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Nota:** la primera vez que ejecutes `detect_logos.py` (o la web), Ultralytics
descarga automáticamente los pesos del modelo (~600MB en total, uso único —
quedan cacheados en la carpeta del proyecto para las siguientes ejecuciones).

## Uso — interfaz web (recomendado)

La forma más sencilla: una página local con botón de subir vídeo, sin usar la
terminal para nada más que arrancar el servidor.

```bash
cd webapp
python app.py
```

Abre `http://127.0.0.1:5000` en el navegador, sube el vídeo y espera (1-2 min
para un clip corto en CPU) — el informe se abre automáticamente al terminar.

## Uso — línea de comandos

```bash
# 1) Detectar logos + trackear en el vídeo
python src/detect_logos.py --video data/raw_clips/mi_clip.mp4

# 2) Calcular métricas de negocio (segundos, rachas, valor mediático)
python src/analytics.py --duration 34.5   # duración del clip en segundos

# 3) Generar el informe HTML final
python src/report.py --duration 34.5 --clip-name "Onboard - Curva 4"
```

El informe queda en `reports/sponsor_roi_report.html` — ábrelo directamente en
un navegador.

## Ajustar qué marcas detectar

Edita `SPONSOR_PROMPTS` en `src/config.py`. No hace falta re-entrenar nada:
al ser detección zero-shot por texto, basta con cambiar los nombres de marca.
Si el modelo detecta poco, baja `CONF_THRESHOLD`; si detecta demasiado ruido,
súbelo.

## Estructura del proyecto

```
src/
  config.py        # marcas a rastrear, umbrales, rutas
  detect_logos.py  # detección + tracking frame a frame -> output/detections.csv
  analytics.py     # detections.csv -> métricas de negocio por marca
  report.py        # métricas -> informe HTML final
webapp/
  app.py           # interfaz web local (subir vídeo -> informe automático)
data/raw_clips/    # coloca aquí los vídeos a analizar
output/            # CSVs intermedios + frames de muestra anotados
reports/           # informe HTML final
STRATEGY.md        # plan técnico de producción + estrategia de negocio/marketing
```
