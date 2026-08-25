# Auditor de ROI para Patrocinadores (Visión Artificial) — Estrategia técnica y de negocio

Este documento acompaña al prototipo funcional en `src/` (detección zero-shot con
YOLO-World + tracking + generación de informe). Cubre tres cosas: cómo pasar del
prototipo a un sistema de producción, cuánto cuesta, y cómo un desarrollador en
solitario consigue la primera reunión con una escudería o una marca patrocinadora.

---

## 1. Del prototipo a producción

### 1.1 Qué hace ya el prototipo (y sus límites)

El pipeline actual usa **YOLO-World**, un detector "open-vocabulary": se le dan
nombres de marca como texto ("Oracle logo", "Petronas logo") y detecta sin haber
visto un solo ejemplo etiquetado de F1. Esto es exactamente lo que necesita un
desarrollador solo para tener *algo funcionando en días, no meses*. Sus límites
honestos:

- Precisión moderada (umbrales de confianza bajos, falsos positivos en fondos
  con texto/formas similares).
- No está adaptado a las condiciones extremas del brief: lluvia, chispas,
  vibración de cámara on-board a 300 km/h, motion blur de adelantamientos.
- La "legibilidad" se aproxima con el % de área del frame que ocupa la bbox —
  un proxy razonable pero no un modelo de percepción humana real.

Esto es intencional: sirve para el vídeo de demo y para validar el concepto
antes de invertir en un dataset propio.

### 1.2 El salto a producción: dataset propio + fine-tuning

Para el nivel de precisión que un patrocinador de 50M€/año va a auditar, hace
falta un modelo **YOLOv8/v9 (o RT-DETR) fine-tuneado** con un dataset propio.

**Pipeline de datos:**

1. **Fuente de vídeo.** El cuello de botella no es técnico, es de derechos: el
   contenido de F1TV es del broadcast oficial (Formula One Management / F1TV),
   protegido por copyright. Para un dataset de entrenamiento interno (no
   redistribuido) esto suele encajar en "investigación/desarrollo de producto"
   bajo un acuerdo con el equipo o la marca cliente, pero **no es libre para
   redistribuir públicamente**. La ruta limpia en producción es: (a) grabaciones
   propias del equipo (onboard, pit wall, boxes — a las que ya tienen derechos),
   o (b) un acuerdo de licencia de clips con FOM/Sky/Canal+ para fines de
   auditoría, algo que agencias como Nielsen Sponsorship o GumGum Sports ya
   tienen negociado. Esto se aborda en la sección 3 (qué mostrar en público vs.
   qué mostrar en una demo privada).
2. **Etiquetado.** Herramientas: CVAT (open-source, autohospedable) o Roboflow
   (SaaS, tiene auto-etiquetado asistido por modelo que acelera mucho esta
   fase). Regla de oro para detección de logos pequeños y con blur: **más vale
   3.000 imágenes bien variadas que 10.000 repetitivas**. Variar:
   - condiciones de luz (sol directo, nublado, anochecer en carreras nocturnas),
   - lluvia y pista mojada (reflejos deforman el logo),
   - ángulos de cámara (onboard, TV principal, helicóptero, pit lane),
   - grados de motion blur (coche a fondo vs. frenada vs. parado en pits),
   - oclusión parcial (otro coche, casco, brazo del piloto, chispas).
3. **Augmentation sintético** para cubrir huecos sin grabar más: motion blur
   direccional, ruido de compresión de vídeo (bitrate bajo de streaming),
   simulación de lluvia/reflejos, recortes a resoluciones de retransmisión
   real (720p/1080p con reescalado agresivo, que es como realmente se ve el
   logo la mayoría del tiempo).
4. **Entrenamiento.** YOLOv8m/l fine-tuneado desde pesos COCO, o partiendo del
   propio checkpoint de YOLO-World para aprovechar el prior de "detección de
   logos" ya aprendido. Con 3-5k imágenes etiquetadas, un fine-tuning razonable
   son 100-300 épocas en una GPU de gama media.

**Coste estimado en la nube (orden de magnitud, revisar precios vigentes):**

| Partida | Opción | Estimación |
|---|---|---|
| Etiquetado asistido | Roboflow Pro | ~$250-500/mes mientras se etiqueta |
| Entrenamiento | AWS `g5.xlarge` (1x A10G) o GCP `a2-highgpu-1g` | $1-1.5/hora, 20-40h total ≈ $30-60 por ciclo de entrenamiento |
| Inferencia por carrera completa (2h vídeo, todas las cámaras) | GPU spot/preemptible | unas pocas horas de cómputo, <$20 por carrera en spot |
| Almacenamiento de clips + dataset | S3 / GCS | Marginal a esta escala (cientos de GB) |

El punto clave para el pitch: **el coste de infraestructura es trivial comparado
con los 50M€/año que paga el patrocinador** — el valor está en los datos y en
la validación del producto, no en el gasto de cómputo.

### 1.3 Mejoras de ingeniería para producción real

- **Tracking multi-cámara**: fusionar detecciones de la señal principal de TV +
  cámaras onboard para no perder exposición cuando la realización cambia de
  plano (el logo "sigue visible" para efectos de contrato aunque la cámara
  cambie, si sigue habiendo un plano del coche).
- **Filtro de legibilidad más fiel**: en vez de solo % de área, combinar
  nitidez (ya está en el prototipo vía varianza del Laplaciano), ángulo de
  visión estimado, y oclusión parcial vs. total.
- **Reconciliación con el guion de realización**: cruzar los timestamps con
  metadatos oficiales de vuelta/sector para frases tipo "vuelta 45" en vez de
  solo segundos absolutos del clip.
- **Validación humana de una muestra**: un revisor humano audita un 5-10% de
  las rachas detectadas por carrera — esto es lo que da credibilidad al
  informe frente al patrocinador y detecta deriva del modelo.

---

## 2. Producto y modelo de negocio

**Cliente:** el departamento de marketing comercial / patrocinios de una
escudería, o directamente la agencia de la marca patrocinadora, o una agencia
de medición de valor mediático (Nielsen Sponsorship, Hookit, GumGum Sports son
"competidores" de facto — también son posibles compradores o socios).

**Propuesta de valor en una frase:** *"Le damos al equipo comercial un número
defendible y por segundo de cuánto se vio cada patrocinador, para que
renegocien el contrato con datos en vez de estimaciones de agencia."*

**Modelo de precio sugerido para arrancar:**
- Informe por carrera (piloto/demo): precio simbólico o gratis a cambio de
  feedback + testimonio, con 1-2 escuderías o marcas.
- Suscripción por temporada una vez validado: por equipo o por marca
  patrocinadora, con dashboard actualizado carrera a carrera.

**Roadmap de producto sugerido:** MVP (este prototipo) → validación con 1
cliente de diseño (mismo dataset, feedback cualitativo) → dataset propio +
fine-tuning → dashboard multi-carrera con comparativa temporada a temporada →
venta a varias escuderías/marcas.

---

## 3. El plan para un desarrollador en solitario: contenido → reunión

La idea no es construir el producto final antes de hablar con nadie. Es
**demostrar en público que el concepto funciona, y usar esa atención para
conseguir una llamada.**

### 3.1 Qué mostrar (y el matiz legal importante)

Grabar un vídeo o hilo mostrando el pipeline detectando logos en un clip de F1
real es la pieza de contenido más persuasiva posible — pero republicar clips
completos del broadcast oficial de F1TV en YouTube/X es contenido con
copyright de Formula One Management, y eso puede generar retirada de
contenido (DMCA) independientemente de la intención. Formas de evitar el
problema sin perder impacto:

- Usar **tu propio vídeo de prueba** (coches de carreras genéricos, karting,
  simuladores tipo iRacing/F1 24 con logos de patrocinador reales en la
  librea, o incluso una maqueta física grabada por ti) para el vídeo público,
  y dejar claro que el modelo se re-entrena igual de rápido sobre clips reales
  de F1 en cuanto haya acceso con licencia.
- Usar un **fragmento muy corto** (segundos, no la carrera completa) bajo
  criterio de uso razonable con comentario/análisis técnico explícito — sigue
  habiendo riesgo, pero es la práctica común en contenido de análisis táctico
  de F1 en YouTube. La opción más segura sigue siendo no depender de ese
  argumento para el pitch.
- Reservar cualquier clip real de F1TV para una **demo privada 1:1** con la
  escudería/marca (uso interno, no publicación), donde el argumento de "fair
  use" no hace falta porque no es contenido público.

Recomendación práctica: construir el vídeo público con datos propios/genéricos
(o simulador), y usar la frase clave "esto mismo funciona sobre vuestras
propias cámaras onboard" como gancho para la reunión, no como algo que ya
mostraste con su contenido.

### 3.2 Estructura del contenido (hilo en X o vídeo corto)

1. El gancho: la cifra que ya vende sola — *"Este patrocinador estuvo 42s
   ininterrumpidos legible en cámara durante una sola batalla. Así es como lo
   medí automáticamente."*
2. 15-30s de pantalla mostrando el pipeline corriendo: el vídeo con las cajas
   de detección en tiempo real, y el informe generándose al final.
3. La captura del informe HTML (exactamente el que genera `src/report.py`) —
   se ve como un producto, no como un script.
4. El cierre: una frase que deja claro que esto es reproducible para
   cualquier equipo/marca, y una llamada a la acción directa ("si eres del
   equipo comercial de una escudería o gestionas patrocinios de F1, escríbeme").
5. Etiquetar/mencionar cuentas relevantes del ecosistema (periodistas de F1
   tech, cuentas de análisis de datos de F1) — no a las escuderías
   directamente en el primer post (leen menos las menciones que reenvíos de
   terceros con autoridad).

### 3.3 De la atención a la reunión

- Buscar en LinkedIn el título "Head of Partnerships" / "Commercial &
  Marketing Director" / "Sponsorship Activation Manager" en las escuderías —
  son las personas que sufren este problema, no el equipo de ingeniería.
- Mensaje directo corto: cifra concreta del demo + enlace al vídeo/hilo +
  una frase ofreciendo correr el mismo análisis sobre un clip que ellos
  elijan, gratis, como prueba de concepto.
- Alternativa más rápida: las agencias de medición de patrocinios (Nielsen
  Sponsorship, Hookit, GumGum Sports, Sponsorlytix) tienen más facilidad de
  acceso y podrían interesarse en licenciar o adquirir la tecnología en vez
  de competir con ella — vale la pena contactarlas en paralelo a las
  escuderías.

---

## 4. Próximos pasos concretos

1. Ejecutar el pipeline sobre el clip de prueba (`src/detect_logos.py` →
   `src/analytics.py` → `src/report.py`) y revisar visualmente los frames de
   muestra en `output/sample_frames/` para calibrar prompts y umbral de
   confianza en `src/config.py`.
2. Grabar el vídeo/hilo de demo siguiendo la sección 3.1 (fuente de vídeo sin
   riesgo de copyright).
3. Publicar y empezar el contacto directo descrito en 3.3, en paralelo.
4. Si hay tracción (respuesta de un equipo o marca), usar esa conversación
   para negociar acceso a clips propios reales y arrancar el dataset de
   fine-tuning descrito en la sección 1.
