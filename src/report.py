"""
Genera el informe HTML final para el equipo de marketing comercial /
patrocinador, a partir de output/summary_by_brand.csv y output/streaks.csv
(y opcionalmente frames de muestra anotados en output/sample_frames/).

Uso:
    python src/report.py --duration 34.5 --clip-name "Onboard - Curva 4"
"""

import argparse
import base64
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR, REPORTS_DIR, ILLUSTRATIVE_USD_PER_SECOND

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def fmt_seconds(s: float) -> str:
    return f"{s:.1f}s"


def fmt_money(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:.0f}"


def img_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def build_bar_chart(summary: pd.DataFrame) -> str:
    """Barra horizontal: segundos totales visibles por marca."""
    if summary.empty:
        return "<p class='muted'>Sin datos.</p>"

    max_val = summary["total_visible_seconds"].max() or 1
    row_h = 40
    chart_h = row_h * len(summary) + 20
    bars = []
    for i, (_, row) in enumerate(summary.iterrows()):
        color = PALETTE[i % len(PALETTE)]
        w_pct = max(row["total_visible_seconds"] / max_val * 100, 2)
        y = i * row_h
        bars.append(f"""
        <div class="bar-row" style="top:{y}px;">
          <div class="bar-label">{row['brand']}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{w_pct:.1f}%; background:{color};"></div>
          </div>
          <div class="bar-value">{fmt_seconds(row['total_visible_seconds'])}</div>
        </div>""")
    return f"""<div class="bar-chart" style="height:{chart_h}px;">{''.join(bars)}</div>"""


def build_timeline(streaks: pd.DataFrame, summary: pd.DataFrame, duration: float) -> str:
    """Timeline horizontal: rachas de visibilidad de cada marca a lo largo del vídeo."""
    if streaks.empty:
        return "<p class='muted'>Sin datos.</p>"

    brands = list(summary["brand"])
    color_map = {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(brands)}
    row_h = 34
    chart_h = row_h * len(brands) + 30

    rows_html = []
    for i, brand in enumerate(brands):
        y = i * row_h
        segs = streaks[streaks["brand"] == brand]
        seg_html = []
        for _, s in segs.iterrows():
            left_pct = max(s["start_t"] / duration * 100, 0)
            width_pct = max((s["end_t"] - s["start_t"]) / duration * 100, 0.4)
            title = f"{brand}: {s['start_t']:.1f}s–{s['end_t']:.1f}s ({s['duration_s']:.1f}s)"
            seg_html.append(
                f'<div class="tl-seg" style="left:{left_pct:.2f}%; width:{width_pct:.2f}%; '
                f'background:{color_map[brand]};" title="{title}"></div>'
            )
        rows_html.append(f"""
        <div class="tl-row" style="top:{y}px;">
          <div class="tl-label">{brand}</div>
          <div class="tl-track">{''.join(seg_html)}</div>
        </div>""")

    ticks = []
    n_ticks = 6
    for t in range(n_ticks + 1):
        sec = duration * t / n_ticks
        left_pct = t / n_ticks * 100
        ticks.append(f'<div class="tl-tick" style="left:{left_pct:.2f}%;">{sec:.0f}s</div>')

    return f"""
    <div class="timeline" style="height:{chart_h}px;">
      {''.join(rows_html)}
    </div>
    <div class="tl-axis">{''.join(ticks)}</div>
    """


def build_stat_tiles(summary: pd.DataFrame, duration: float) -> str:
    if summary.empty:
        return ""
    top = summary.iloc[0]
    total_visible_any = summary["total_visible_seconds"].sum()
    total_value = summary["illustrative_media_value_usd"].sum()

    tiles = [
        ("Racha ininterrumpida más larga",
         fmt_seconds(top["longest_streak_seconds"]),
         f"{top['brand']} · {top['longest_streak_window']}"),
        ("Exposición total (marca líder)",
         fmt_seconds(top["total_visible_seconds"]),
         f"{top['pct_of_video']}% del clip analizado"),
        ("Valor mediático ilustrativo",
         fmt_money(total_value),
         f"a ${ILLUSTRATIVE_USD_PER_SECOND:,}/s · cifra de ejemplo, no oficial"),
        ("Duración del clip analizado",
         fmt_seconds(duration),
         f"{len(summary)} marca(s) detectada(s)"),
    ]
    html = []
    for label, value, sub in tiles:
        html.append(f"""
        <div class="tile">
          <div class="tile-label">{label}</div>
          <div class="tile-value">{value}</div>
          <div class="tile-sub">{sub}</div>
        </div>""")
    return "".join(html)


def build_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return ""
    rows = []
    for _, r in summary.iterrows():
        rows.append(f"""
        <tr>
          <td>{r['brand']}</td>
          <td class="num">{fmt_seconds(r['total_visible_seconds'])}</td>
          <td class="num">{r['pct_of_video']}%</td>
          <td class="num">{r['n_appearances']}</td>
          <td class="num">{fmt_seconds(r['longest_streak_seconds'])}</td>
          <td class="num">{r['legible_ratio_pct']}%</td>
          <td class="num">{r['avg_confidence']}</td>
          <td class="num">{fmt_money(r['illustrative_media_value_usd'])}</td>
        </tr>""")
    return "".join(rows)


def build_gallery(sample_dir: Path, max_images: int = 6) -> str:
    if not sample_dir.exists():
        return ""
    images = sorted(sample_dir.glob("*.jpg"))[:max_images]
    if not images:
        return ""
    cards = []
    for img in images:
        uri = img_to_data_uri(img)
        cards.append(f'<div class="gallery-card"><img src="{uri}" alt="frame detectado" /></div>')
    return f'<div class="gallery">{"".join(cards)}</div>'


HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Informe de Exposición de Patrocinador</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --baseline: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --accent: #2a78d6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --baseline: #383835;
      --border: rgba(255,255,255,0.10);
      --accent: #3987e5;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --accent: #3987e5;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--page); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  .viz-root {{ max-width: 980px; margin: 0 auto; padding: 40px 24px 80px; color: var(--text-primary); }}
  header.report-header {{ margin-bottom: 32px; }}
  .eyebrow {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 6px; }}
  h1 {{ font-size: 28px; margin: 0 0 8px; font-weight: 650; }}
  .subtitle {{ color: var(--text-secondary); font-size: 15px; margin: 0; }}
  section {{ margin-top: 40px; }}
  h2 {{ font-size: 17px; font-weight: 650; margin: 0 0 16px; }}
  .muted {{ color: var(--muted); font-size: 14px; }}

  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
  .tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }}
  .tile-label {{ font-size: 13px; color: var(--text-secondary); margin-bottom: 8px; }}
  .tile-value {{ font-size: 30px; font-weight: 650; line-height: 1.1; }}
  .tile-sub {{ font-size: 12.5px; color: var(--muted); margin-top: 6px; }}

  .bar-chart {{ position: relative; }}
  .bar-row {{ position: absolute; left: 0; right: 0; display: grid; grid-template-columns: 140px 1fr 64px; align-items: center; gap: 12px; height: 40px; }}
  .bar-label {{ font-size: 13.5px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .bar-track {{ background: var(--grid); border-radius: 4px; height: 20px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .bar-value {{ font-size: 13.5px; color: var(--text-primary); text-align: right; font-variant-numeric: tabular-nums; }}

  .timeline {{ position: relative; margin-bottom: 8px; }}
  .tl-row {{ position: absolute; left: 0; right: 0; display: grid; grid-template-columns: 140px 1fr; align-items: center; gap: 12px; height: 34px; }}
  .tl-label {{ font-size: 13.5px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .tl-track {{ position: relative; background: var(--grid); border-radius: 4px; height: 14px; }}
  .tl-seg {{ position: absolute; top: 0; bottom: 0; border-radius: 3px; }}
  .tl-axis {{ position: relative; height: 20px; margin-left: 152px; border-top: 1px solid var(--baseline); }}
  .tl-tick {{ position: absolute; top: 4px; transform: translateX(-50%); font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--grid); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

  .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
  .gallery-card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
  .gallery-card img {{ width: 100%; display: block; }}

  footer {{ margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--grid); color: var(--muted); font-size: 12px; line-height: 1.6; }}
</style>
<div class="viz-root">
  <header class="report-header">
    <p class="eyebrow">Informe de exposición en pantalla &middot; visión artificial</p>
    <h1>Auditoría de ROI de Patrocinador</h1>
    <p class="subtitle">{clip_name} &middot; generado el {gen_date}</p>
  </header>

  <section>
    <h2>Resumen</h2>
    <div class="tiles">{stat_tiles}</div>
  </section>

  <section>
    <h2>Segundos totales de exposición legible por marca</h2>
    {bar_chart}
  </section>

  <section>
    <h2>Línea temporal de apariciones en el clip</h2>
    {timeline}
  </section>

  <section>
    <h2>Detalle por marca</h2>
    <table>
      <thead>
        <tr>
          <th>Marca</th>
          <th class="num">Seg. visibles</th>
          <th class="num">% del clip</th>
          <th class="num">Apariciones</th>
          <th class="num">Racha máx.</th>
          <th class="num">% legible</th>
          <th class="num">Confianza media</th>
          <th class="num">Valor mediático*</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </section>

  {gallery_section}

  <footer>
    Generado automáticamente por el pipeline de Auditoría de ROI de Patrocinadores (YOLO-World zero-shot
    + ByteTrack). Prototipo de demostración: detección por texto sin fine-tuning, pensado para validar el
    concepto rápidamente. *El valor mediático es una cifra ilustrativa (${usd_per_sec}/segundo) para fines
    de demo &mdash; en un informe real se sustituye por datos de la agencia de medición de audiencia del
    equipo (Nielsen Sponsorship, Hookit, GumGum Sports, etc.). Para precisión de nivel producción bajo
    lluvia, chispas y vibración de cámara a 300&nbsp;km/h, ver STRATEGY.md para el plan de fine-tuning con
    dataset propio.
  </footer>
</div>
"""


def run(duration: float, clip_name: str = "Clip analizado"):
    import datetime

    summary_path = OUTPUT_DIR / "summary_by_brand.csv"
    streaks_path = OUTPUT_DIR / "streaks.csv"
    sample_dir = OUTPUT_DIR / "sample_frames"

    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    streaks = pd.read_csv(streaks_path) if streaks_path.exists() else pd.DataFrame()

    html = HTML_TEMPLATE.format(
        clip_name=clip_name,
        gen_date=datetime.date.today().isoformat(),
        stat_tiles=build_stat_tiles(summary, duration),
        bar_chart=build_bar_chart(summary),
        timeline=build_timeline(streaks, summary, duration),
        table_rows=build_table(summary),
        gallery_section=(
            f"<section><h2>Frames de muestra detectados</h2>{build_gallery(sample_dir)}</section>"
            if any(sample_dir.glob('*.jpg')) else ""
        ),
        usd_per_sec=f"{ILLUSTRATIVE_USD_PER_SECOND:,}",
    )

    out_path = REPORTS_DIR / "sponsor_roi_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Informe generado: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--clip-name", default="Clip analizado")
    args = parser.parse_args()
    run(args.duration, args.clip_name)
