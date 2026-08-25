"""
Motor de analítica: convierte output/detections.csv (detecciones crudas
frame a frame) en métricas de negocio por marca:

  - segundos totales de exposición
  - racha continua más larga (ej. "42 segundos ininterrumpidos")
  - nº de apariciones (entradas/salidas de plano)
  - % de esas apariciones que son "legibles" (bbox suficientemente grande)
  - valor mediático equivalente ilustrativo (USD)

La "racha continua" se calcula por track_id, uniendo detecciones del mismo
track aunque haya micro-huecos (<= MAX_GAP_SECONDS) causados por motion
blur, chispas u oclusión momentánea (otro coche adelantando, etc.).
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    OUTPUT_DIR,
    MAX_GAP_SECONDS,
    MIN_AREA_FRACTION_LEGIBLE,
    ILLUSTRATIVE_USD_PER_SECOND,
)


@dataclass
class Streak:
    brand: str
    track_id: int
    start_t: float
    end_t: float
    n_detections: int
    legible_detections: int
    avg_conf: float
    max_area_fraction: float

    @property
    def duration(self) -> float:
        return round(self.end_t - self.start_t, 2)


def build_streaks(df: pd.DataFrame) -> list[Streak]:
    """Agrupa detecciones consecutivas del mismo track_id en 'rachas' continuas,
    tolerando huecos cortos (motion blur / chispas / oclusión momentánea)."""
    streaks: list[Streak] = []

    for (brand, track_id), group in df.groupby(["brand", "track_id"]):
        group = group.sort_values("t_seconds")
        current = None

        for _, row in group.iterrows():
            t = row["t_seconds"]
            is_legible = row["area_fraction"] >= MIN_AREA_FRACTION_LEGIBLE

            if current is None:
                current = {
                    "start_t": t, "end_t": t, "confs": [row["conf"]],
                    "n": 1, "legible": int(is_legible),
                    "max_area": row["area_fraction"],
                }
                continue

            gap = t - current["end_t"]
            if gap <= MAX_GAP_SECONDS:
                current["end_t"] = t
                current["confs"].append(row["conf"])
                current["n"] += 1
                current["legible"] += int(is_legible)
                current["max_area"] = max(current["max_area"], row["area_fraction"])
            else:
                streaks.append(Streak(
                    brand=brand, track_id=int(track_id),
                    start_t=current["start_t"], end_t=current["end_t"],
                    n_detections=current["n"], legible_detections=current["legible"],
                    avg_conf=sum(current["confs"]) / len(current["confs"]),
                    max_area_fraction=current["max_area"],
                ))
                current = {
                    "start_t": t, "end_t": t, "confs": [row["conf"]],
                    "n": 1, "legible": int(is_legible),
                    "max_area": row["area_fraction"],
                }

        if current is not None:
            streaks.append(Streak(
                brand=brand, track_id=int(track_id),
                start_t=current["start_t"], end_t=current["end_t"],
                n_detections=current["n"], legible_detections=current["legible"],
                avg_conf=sum(current["confs"]) / len(current["confs"]),
                max_area_fraction=current["max_area"],
            ))

    return streaks


def summarize_by_brand(streaks: list[Streak], video_duration_s: float) -> pd.DataFrame:
    rows = []
    by_brand: dict[str, list[Streak]] = {}
    for s in streaks:
        by_brand.setdefault(s.brand, []).append(s)

    for brand, brand_streaks in by_brand.items():
        total_visible = sum(s.duration for s in brand_streaks)
        longest = max(brand_streaks, key=lambda s: s.duration)
        legible_ratio = (
            sum(s.legible_detections for s in brand_streaks)
            / max(sum(s.n_detections for s in brand_streaks), 1)
        )
        rows.append({
            "brand": brand,
            "total_visible_seconds": round(total_visible, 2),
            "pct_of_video": round(100 * total_visible / max(video_duration_s, 1e-6), 1),
            "n_appearances": len(brand_streaks),
            "longest_streak_seconds": longest.duration,
            "longest_streak_window": f"{longest.start_t:.1f}s - {longest.end_t:.1f}s",
            "legible_ratio_pct": round(100 * legible_ratio, 1),
            "avg_confidence": round(sum(s.avg_conf for s in brand_streaks) / len(brand_streaks), 2),
            "illustrative_media_value_usd": round(total_visible * ILLUSTRATIVE_USD_PER_SECOND),
        })

    out = pd.DataFrame(rows).sort_values("total_visible_seconds", ascending=False)
    return out.reset_index(drop=True)


def run(detections_csv: Path, video_duration_s: float):
    df = pd.read_csv(detections_csv)
    if df.empty:
        print("No hay detecciones en el CSV. Revisa umbral de confianza / prompts en config.py.")
        return None, None

    streaks = build_streaks(df)
    summary = summarize_by_brand(streaks, video_duration_s)

    streaks_df = pd.DataFrame([{
        "brand": s.brand, "track_id": s.track_id,
        "start_t": s.start_t, "end_t": s.end_t, "duration_s": s.duration,
        "n_detections": s.n_detections, "legible_detections": s.legible_detections,
        "avg_conf": round(s.avg_conf, 2), "max_area_fraction": s.max_area_fraction,
    } for s in streaks]).sort_values("duration_s", ascending=False)

    summary_path = OUTPUT_DIR / "summary_by_brand.csv"
    streaks_path = OUTPUT_DIR / "streaks.csv"
    summary.to_csv(summary_path, index=False)
    streaks_df.to_csv(streaks_path, index=False)

    print(f"Resumen por marca guardado en: {summary_path}")
    print(f"Rachas individuales guardadas en: {streaks_path}")
    print("\n=== RESUMEN ===")
    print(summary.to_string(index=False))

    return summary, streaks_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", default=str(OUTPUT_DIR / "detections.csv"))
    parser.add_argument("--duration", type=float, required=True, help="Duración del vídeo en segundos.")
    args = parser.parse_args()
    run(Path(args.detections), args.duration)
