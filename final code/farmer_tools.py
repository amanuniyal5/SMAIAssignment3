"""
farmer_tools.py
───────────────
Standalone utilities that extend app.py with real farmer-facing features.
Every function here is imported by app.py — nothing breaks if this file
is missing (all imports are guarded).

Modules:
  1. FieldDiary     — SQLite-backed scan history per device session
  2. WeatherRisk    — disease spread risk from Open-Meteo (free, no key)
  3. BatchAnalyser  — multi-leaf upload → field-level verdict
  4. ReportExporter — PDF/PNG shareable report generation
  5. Translations   — Hindi + regional language strings
  6. TreatmentCosts — per-disease chemical costs and availability info
"""

import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ══════════════════════════════════════════════════════════════════
# 1. FIELD DIARY — scan history stored in local SQLite
# ══════════════════════════════════════════════════════════════════
DIARY_DB = Path("field_diary.db")

class FieldDiary:
    """
    Persists every scan as a record: image thumbnail, prediction,
    confidence, GPS (if provided), crop, date.
    Lets a farmer track disease progression over the season.
    """

    def __init__(self, db_path: str = str(DIARY_DB)):
        self.db = Path(db_path)
        self._init()

    def _conn(self):
        return sqlite3.connect(self.db)

    def _init(self):
        with self._conn() as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                scan_id     TEXT PRIMARY KEY,
                timestamp   TEXT,
                crop        TEXT,
                disease     TEXT,
                display_name TEXT,
                confidence  REAL,
                severity    TEXT,
                action      TEXT,
                lat         REAL,
                lon         REAL,
                location_name TEXT,
                thumbnail   BLOB,
                notes       TEXT,
                shared      INTEGER DEFAULT 0
            )""")

    def save_scan(self, crop: str, disease: str, display_name: str,
                  confidence: float, severity: str, action: str,
                  img_pil: Image.Image, notes: str = "",
                  lat: float = None, lon: float = None,
                  location_name: str = "") -> str:
        scan_id = str(uuid.uuid4())[:8]
        ts      = datetime.now().isoformat(timespec="seconds")

        # Make 120×120 thumbnail for storage
        thumb = img_pil.convert("RGB").resize((120, 120))
        buf   = BytesIO()
        thumb.save(buf, "JPEG", quality=75)
        thumb_bytes = buf.getvalue()

        with self._conn() as con:
            con.execute("""
            INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (scan_id, ts, crop, disease, display_name, confidence,
                  severity, action, lat, lon, location_name,
                  thumb_bytes, notes, 0))
        return scan_id

    def get_all(self) -> list[dict]:
        with self._conn() as con:
            cur = con.execute(
                "SELECT * FROM scans ORDER BY timestamp DESC")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            if d.get("thumbnail"):
                try:
                    d["thumbnail_pil"] = Image.open(BytesIO(d["thumbnail"]))
                except Exception:
                    d["thumbnail_pil"] = None
            result.append(d)
        return result

    def get_recent(self, n: int = 10) -> list[dict]:
        return self.get_all()[:n]

    def get_by_crop(self, crop: str) -> list[dict]:
        return [r for r in self.get_all()
                if r["crop"].lower() == crop.lower()]

    def delete(self, scan_id: str):
        with self._conn() as con:
            con.execute("DELETE FROM scans WHERE scan_id=?", (scan_id,))

    def summary_stats(self) -> dict:
        scans = self.get_all()
        if not scans:
            return {}
        from collections import Counter
        crops     = Counter(s["crop"] for s in scans)
        diseases  = Counter(s["disease"] for s in scans if s["disease"] != "None")
        severities= Counter(s["severity"] for s in scans)
        return {
            "total_scans":    len(scans),
            "unique_crops":   len(crops),
            "top_crop":       crops.most_common(1)[0] if crops else None,
            "top_disease":    diseases.most_common(1)[0] if diseases else None,
            "severity_counts":dict(severities),
            "healthy_pct":    round(sum(1 for s in scans if s["severity"]=="none")/len(scans)*100, 1),
        }

    def export_csv(self) -> str:
        """Returns CSV string of all scans (no thumbnails)."""
        scans = self.get_all()
        if not scans:
            return "No scans recorded yet."
        cols = ["scan_id","timestamp","crop","disease","display_name",
                "confidence","severity","location_name","notes"]
        lines = [",".join(cols)]
        for s in scans:
            lines.append(",".join(str(s.get(c,"")) for c in cols))
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 2. WEATHER RISK — Open-Meteo API (free, no key required)
# ══════════════════════════════════════════════════════════════════
DISEASE_WEATHER_CONDITIONS = {
    # disease_key_fragment → {"temp_range": (min,max), "humidity_min": pct,
    #                          "rain_risk": bool, "risk_label": str}
    "late_blight":        {"temp_range":(10,20), "humidity_min":90, "rain_risk":True,
                           "risk_label":"Late Blight thrives in cool, wet conditions"},
    "early_blight":       {"temp_range":(24,29), "humidity_min":70, "rain_risk":True,
                           "risk_label":"Early Blight peaks in warm, humid weather"},
    "powdery_mildew":     {"temp_range":(20,27), "humidity_min":50, "rain_risk":False,
                           "risk_label":"Powdery Mildew thrives in warm, dry days + humid nights"},
    "bacterial_spot":     {"temp_range":(24,30), "humidity_min":80, "rain_risk":True,
                           "risk_label":"Bacterial Spot spreads rapidly via rain splash"},
    "black_rot":          {"temp_range":(20,30), "humidity_min":75, "rain_risk":True,
                           "risk_label":"Black Rot spreads during warm, wet periods"},
    "gray_leaf_spot":     {"temp_range":(25,30), "humidity_min":85, "rain_risk":True,
                           "risk_label":"Gray Leaf Spot thrives in warm, humid nights"},
    "northern_leaf_blight":{"temp_range":(18,27),"humidity_min":80, "rain_risk":True,
                            "risk_label":"Northern Leaf Blight peaks in moderate, humid weather"},
    "mosaic_virus":       {"temp_range":(20,32), "humidity_min":40, "rain_risk":False,
                           "risk_label":"Mosaic Virus spreads via aphid vectors in warm, dry conditions"},
    "yellow_leaf_curl":   {"temp_range":(25,35), "humidity_min":40, "rain_risk":False,
                           "risk_label":"TYLCV spreads via whitefly in hot, dry conditions"},
    "leaf_scorch":        {"temp_range":(20,28), "humidity_min":75, "rain_risk":True,
                           "risk_label":"Leaf Scorch peaks during wet, warm periods"},
    "rust":               {"temp_range":(15,25), "humidity_min":80, "rain_risk":True,
                           "risk_label":"Rust diseases spread in moderate, wet conditions"},
    "spider_mites":       {"temp_range":(27,35), "humidity_min":20, "rain_risk":False,
                           "risk_label":"Spider Mites explode in hot, dry weather"},
    "haunglongbing":      {"temp_range":(25,32), "humidity_min":50, "rain_risk":False,
                           "risk_label":"HLB spreads via psyllid insects year-round in warm climates"},
}

def get_weather(lat: float, lon: float) -> dict | None:
    """
    Fetches current weather from Open-Meteo (free, no API key).
    Returns temperature (°C), relative humidity (%), precipitation (mm),
    wind speed, weather code.
    """
    try:
        import urllib.request
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m,"
               f"precipitation,wind_speed_10m,weather_code"
               f"&timezone=auto&forecast_days=1")
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        curr = data["current"]
        return {
            "temp_c":    curr["temperature_2m"],
            "humidity":  curr["relative_humidity_2m"],
            "precip_mm": curr["precipitation"],
            "wind_kmh":  curr["wind_speed_10m"],
            "code":      curr["weather_code"],
            "lat": lat, "lon": lon,
        }
    except Exception:
        return None

def assess_spread_risk(disease_class: str, weather: dict) -> dict:
    """
    Given a detected disease and current weather, returns:
      - risk_level: low / moderate / high / critical
      - risk_score: 0-100
      - reason: human-readable explanation
      - spray_today: bool — should farmer spray today?
    """
    if not weather:
        return {"risk_level": "unknown", "risk_score": 0,
                "reason": "Weather data unavailable",
                "spray_today": False}

    disease_lower = disease_class.lower().replace("___","_").replace(" ","_")
    matched_cond  = None
    for key, cond in DISEASE_WEATHER_CONDITIONS.items():
        if key in disease_lower:
            matched_cond = cond
            break

    if not matched_cond:
        return {"risk_level": "moderate", "risk_score": 40,
                "reason": "No specific weather model for this disease.",
                "spray_today": False}

    t   = weather["temp_c"]
    rh  = weather["humidity"]
    pr  = weather["precip_mm"]
    t_min, t_max = matched_cond["temp_range"]

    score = 0
    reasons = []

    # Temperature match
    if t_min <= t <= t_max:
        score += 40
        reasons.append(f"Temperature {t:.1f}°C is in the danger range ({t_min}–{t_max}°C)")
    elif abs(t - (t_min+t_max)/2) < 5:
        score += 20
        reasons.append(f"Temperature {t:.1f}°C is near the danger range")

    # Humidity
    if rh >= matched_cond["humidity_min"]:
        score += 35
        reasons.append(f"Humidity {rh}% is above the risk threshold ({matched_cond['humidity_min']}%)")
    elif rh >= matched_cond["humidity_min"] - 10:
        score += 15

    # Rain
    if matched_cond["rain_risk"] and pr > 0.5:
        score += 25
        reasons.append(f"Recent rainfall ({pr:.1f}mm) will spread spores via splash")
    elif not matched_cond["rain_risk"] and pr < 1:
        score += 15
        reasons.append(f"Dry conditions favour {disease_class} spread")

    risk_level = ("critical" if score >= 80 else
                  "high"     if score >= 55 else
                  "moderate" if score >= 30 else "low")

    spray = score >= 55

    reason_str = matched_cond["risk_label"] + ". " + ". ".join(reasons) + "."

    return {
        "risk_level": risk_level,
        "risk_score": min(score, 100),
        "reason":     reason_str,
        "spray_today": spray,
    }


# ══════════════════════════════════════════════════════════════════
# 3. BATCH ANALYSER — multi-leaf field scan
# ══════════════════════════════════════════════════════════════════
def analyse_batch(images: list[Image.Image], run_inference_fn,
                  disease_info: dict, lookup_fn) -> dict:
    """
    Runs inference on multiple leaf images and aggregates:
    - Majority disease (most common top-prediction)
    - Field infection rate (% of leaves diseased)
    - Per-image results
    - Recommendation based on infection rate
    """
    results = []
    disease_counts = {}

    for i, img in enumerate(images):
        try:
            preds = run_inference_fn(img)
            top_class, top_conf = preds[0]
            info, _ = lookup_fn(top_class, disease_info)
            severity = info.get("severity", "moderate")
            results.append({
                "index":        i + 1,
                "top_class":    top_class,
                "display_name": info.get("display_name", top_class),
                "confidence":   top_conf,
                "severity":     severity,
                "is_diseased":  severity != "none",
                "action":       info.get("action", ""),
                "emoji":        info.get("emoji", "🌿"),
            })
            d = info.get("disease", top_class)
            disease_counts[d] = disease_counts.get(d, 0) + 1
        except Exception as e:
            results.append({"index": i+1, "error": str(e)})

    n_total    = len(results)
    n_diseased = sum(1 for r in results if r.get("is_diseased", False))
    infect_pct = round(n_diseased / n_total * 100, 1) if n_total > 0 else 0

    # Most common disease
    if disease_counts:
        major_disease = max(disease_counts, key=disease_counts.get)
    else:
        major_disease = "Unknown"

    # Field-level recommendation
    if infect_pct == 0:
        field_verdict  = "✅ Field appears healthy"
        urgency        = "none"
        field_action   = "Continue routine monitoring. No immediate intervention needed."
    elif infect_pct < 25:
        field_verdict  = "⚠️ Early infection detected"
        urgency        = "moderate"
        field_action   = (f"About {infect_pct}% of sampled leaves show signs of {major_disease}. "
                          f"Apply targeted treatment to affected areas and monitor closely for spread.")
    elif infect_pct < 60:
        field_verdict  = "🔴 Moderate-to-severe field infection"
        urgency        = "high"
        field_action   = (f"{infect_pct}% of leaves are infected with {major_disease}. "
                          f"Immediate field-wide treatment is required. Consider consulting your "
                          f"local Krishi Vigyan Kendra for emergency support.")
    else:
        field_verdict  = "🟣 CRITICAL — widespread infection"
        urgency        = "critical"
        field_action   = (f"Over {infect_pct}% of leaves show {major_disease}. "
                          f"The field is severely compromised. Act today — contact your "
                          f"KVK immediately and begin emergency spray program.")

    return {
        "n_images":       n_total,
        "n_diseased":     n_diseased,
        "infection_pct":  infect_pct,
        "major_disease":  major_disease,
        "disease_counts": disease_counts,
        "field_verdict":  field_verdict,
        "urgency":        urgency,
        "field_action":   field_action,
        "per_image":      results,
    }


# ══════════════════════════════════════════════════════════════════
# 4. REPORT EXPORTER — shareable PNG card
# ══════════════════════════════════════════════════════════════════
def generate_report_image(
    leaf_img: Image.Image,
    display_name: str,
    confidence: float,
    severity: str,
    description: str,
    action: str,
    crop: str,
    farmer_name: str = "",
    field_name: str = "",
    location: str = "",
    weather_risk: dict = None,
) -> Image.Image:
    """
    Generates a clean, shareable report card (PNG) — like a
    WhatsApp-shareable diagnosis card a farmer can send to their agronomist.
    No external fonts required; falls back to PIL default.
    """
    W, H = 800, 600
    canvas = Image.new("RGB", (W, H), "#f8fafc")
    draw   = ImageDraw.Draw(canvas)

    # ── Try to load a nice font, fall back gracefully ──
    def font(size):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            try:
                return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
            except Exception:
                return ImageFont.load_default()

    # Header band
    sev_colors = {"none":"#22c55e","moderate":"#f59e0b",
                  "high":"#ef4444","critical":"#7c3aed"}
    hdr_color  = sev_colors.get(severity, "#6b7280")
    draw.rectangle([0, 0, W, 90], fill=hdr_color)
    draw.text((20, 14), "🌿 CropGuard AI — Disease Report", font=font(22), fill="white")
    draw.text((20, 52), datetime.now().strftime("%d %B %Y, %I:%M %p"), font=font(14), fill="#d1fae5")

    # Leaf thumbnail (left column)
    thumb = leaf_img.convert("RGB").resize((180, 180))
    canvas.paste(thumb, (20, 105))

    # Draw border around thumbnail
    draw.rectangle([19, 104, 201, 286], outline="#e5e7eb", width=2)

    # Right column — diagnosis
    rx = 215
    draw.text((rx, 105), display_name, font=font(20), fill="#111827")

    sev_label = {"none":"HEALTHY","moderate":"MODERATE RISK",
                 "high":"HIGH RISK","critical":"CRITICAL"}.get(severity, severity.upper())
    draw.rounded_rectangle([rx, 132, rx+160, 155], radius=10,
                            fill=hdr_color)
    draw.text((rx+10, 136), sev_label, font=font(14), fill="white")

    draw.text((rx, 165), f"Crop: {crop}", font=font(14), fill="#374151")
    draw.text((rx, 185), f"Confidence: {confidence*100:.1f}%", font=font(14), fill="#374151")

    if farmer_name:
        draw.text((rx, 205), f"Farmer: {farmer_name}", font=font(13), fill="#6b7280")
    if field_name:
        draw.text((rx, 223), f"Field: {field_name}", font=font(13), fill="#6b7280")
    if location:
        draw.text((rx, 241), f"Location: {location}", font=font(13), fill="#6b7280")

    # Description (wrapped)
    y_text = 295
    draw.text((20, y_text - 22), "Diagnosis:", font=font(15), fill="#111827")
    words = description.split()
    line, lines = [], []
    for w in words:
        test = " ".join(line + [w])
        if draw.textlength(test, font=font(13)) < W - 40:
            line.append(w)
        else:
            lines.append(" ".join(line))
            line = [w]
    if line: lines.append(" ".join(line))
    for l in lines[:4]:
        draw.text((20, y_text), l, font=font(13), fill="#374151")
        y_text += 18

    # Action box
    y_act = y_text + 10
    draw.rounded_rectangle([16, y_act, W-16, y_act+90], radius=8,
                            fill="#f0fdf4", outline="#86efac", width=1)
    draw.text((26, y_act+8), "✅ Recommended Action:", font=font(14), fill="#166534")
    words2 = action.split()
    line2, lines2 = [], []
    for w in words2:
        test = " ".join(line2 + [w])
        if draw.textlength(test, font=font(12)) < W - 60:
            line2.append(w)
        else:
            lines2.append(" ".join(line2))
            line2 = [w]
    if line2: lines2.append(" ".join(line2))
    ay = y_act + 28
    for l in lines2[:3]:
        draw.text((26, ay), l, font=font(12), fill="#15803d")
        ay += 17

    # Weather risk strip (if available)
    if weather_risk and weather_risk.get("risk_level") != "unknown":
        wr_y = H - 55
        wr_colors = {"low":"#22c55e","moderate":"#f59e0b",
                     "high":"#ef4444","critical":"#7c3aed"}
        wr_col = wr_colors.get(weather_risk["risk_level"], "#6b7280")
        draw.rectangle([0, wr_y - 5, W, H], fill="#1e293b")
        score = weather_risk.get("risk_score", 0)
        spray = "⚠️ SPRAY TODAY" if weather_risk.get("spray_today") else "Monitor closely"
        draw.text((20, wr_y + 2),
                  f"🌦 Spread Risk: {weather_risk['risk_level'].upper()} ({score}/100)  —  {spray}",
                  font=font(13), fill=wr_col)

    # Footer
    draw.text((W//2 - 160, H - 22),
              "Generated by CropGuard AI · PlantVillage Model",
              font=font(11), fill="#9ca3af")

    return canvas


def export_report_bytes(report_img: Image.Image) -> bytes:
    """Returns PNG bytes for download."""
    buf = BytesIO()
    report_img.save(buf, "PNG", dpi=(150, 150))
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════
# 5. TREATMENT COSTS — per-disease chemicals, cost ranges, sources
# ══════════════════════════════════════════════════════════════════
TREATMENT_DB = {
    "Apple Scab": {
        "chemicals": [
            {"name": "Mancozeb 75% WP",      "dose": "2.5 g/L water",  "cost_per_kg": "₹180–250",  "applications": 3},
            {"name": "Captan 50% WP",         "dose": "2.5 g/L water",  "cost_per_kg": "₹350–450",  "applications": 3},
            {"name": "Myclobutanil 10% WP",   "dose": "1 g/L water",    "cost_per_kg": "₹1200–1800","applications": 2},
        ],
        "organic": "Neem oil (5ml/L) + Copper oxychloride (3g/L)",
        "where_to_buy": "Krishi seva kendra, local agri-input dealers",
        "cost_estimate_per_acre": "₹800–1,500",
        "critical_timing": "Apply before rain forecast",
    },
    "Late Blight": {
        "chemicals": [
            {"name": "Cymoxanil + Mancozeb",   "dose": "3 g/L water",   "cost_per_kg": "₹500–700",  "applications": 3},
            {"name": "Metalaxyl + Mancozeb",   "dose": "2.5 g/L water", "cost_per_kg": "₹650–900",  "applications": 3},
            {"name": "Fenamidone + Mancozeb",  "dose": "2 g/L water",   "cost_per_kg": "₹900–1200", "applications": 2},
        ],
        "organic": "Copper hydroxide (3g/L). Bordeaux mixture (1%)",
        "where_to_buy": "Any registered pesticide dealer. EMERGENCY — buy today.",
        "cost_estimate_per_acre": "₹1,200–2,500",
        "critical_timing": "IMMEDIATE — do not delay even one day",
    },
    "Early Blight": {
        "chemicals": [
            {"name": "Mancozeb 75% WP",       "dose": "2.5 g/L water",  "cost_per_kg": "₹180–250",  "applications": 4},
            {"name": "Chlorothalonil 75% WP",  "dose": "2 g/L water",    "cost_per_kg": "₹280–380",  "applications": 3},
            {"name": "Azoxystrobin 23% SC",    "dose": "1 ml/L water",   "cost_per_litre": "₹2500–3500","applications": 2},
        ],
        "organic": "Neem oil (5ml/L). Trichoderma viride (5g/L)",
        "where_to_buy": "Krishi seva kendra, online: BigHaat, AgroStar",
        "cost_estimate_per_acre": "₹600–1,200",
        "critical_timing": "Apply at first sign of spots",
    },
    "Powdery Mildew": {
        "chemicals": [
            {"name": "Sulfur 80% WP",          "dose": "3 g/L water",    "cost_per_kg": "₹120–180",  "applications": 3},
            {"name": "Myclobutanil 10% WP",    "dose": "1 g/L water",    "cost_per_kg": "₹1200–1800","applications": 2},
            {"name": "Potassium Bicarbonate",  "dose": "5 g/L water",    "cost_per_kg": "₹200–350",  "applications": 4},
        ],
        "organic": "Baking soda (5g/L) + neem oil (5ml/L). Whey solution (1:10).",
        "where_to_buy": "Agri shops, BigHaat app, DeHaat app",
        "cost_estimate_per_acre": "₹400–900",
        "critical_timing": "Before disease covers >10% leaf area",
    },
    "Bacterial Spot": {
        "chemicals": [
            {"name": "Copper Hydroxide 77% WP","dose": "3 g/L water",    "cost_per_kg": "₹350–550",  "applications": 5},
            {"name": "Copper Oxychloride 50%", "dose": "3 g/L water",    "cost_per_kg": "₹220–320",  "applications": 5},
            {"name": "Streptomycin Sulfate",   "dose": "0.5 g/L water",  "cost_per_g": "₹8–12",      "applications": 3},
        ],
        "organic": "Copper soap spray (2%). Neem seed kernel extract (5%).",
        "where_to_buy": "Licensed pesticide shops only (Streptomycin requires prescription in some states)",
        "cost_estimate_per_acre": "₹700–1,400",
        "critical_timing": "Begin at first lesion. Spray every 5–7 days during wet weather.",
    },
    "Black Rot": {
        "chemicals": [
            {"name": "Mancozeb 75% WP",        "dose": "2.5 g/L water",  "cost_per_kg": "₹180–250", "applications": 4},
            {"name": "Myclobutanil 10% WP",    "dose": "1 g/L water",    "cost_per_kg": "₹1200–1800","applications": 2},
            {"name": "Thiophanate-methyl",     "dose": "1.5 g/L water",  "cost_per_kg": "₹600–900", "applications": 3},
        ],
        "organic": "Remove all infected material first. Bordeaux mixture (1%).",
        "where_to_buy": "Agri input centres, Kisan Call Centre: 1800-180-1551",
        "cost_estimate_per_acre": "₹800–1,600",
        "critical_timing": "Remove mummified fruit first, then spray",
    },
    "Gray Leaf Spot": {
        "chemicals": [
            {"name": "Propiconazole 25% EC",   "dose": "1 ml/L water",   "cost_per_litre": "₹700–1000","applications": 2},
            {"name": "Azoxystrobin 23% SC",    "dose": "1 ml/L water",   "cost_per_litre": "₹2500–3500","applications": 2},
            {"name": "Mancozeb 75% WP",        "dose": "2.5 g/L water",  "cost_per_kg": "₹180–250",  "applications": 3},
        ],
        "organic": "Trichoderma-based biocontrol. Crop rotation is most effective.",
        "where_to_buy": "Agri shops. Biocontrol agents: National Agri Shops, Biopesticide vendors",
        "cost_estimate_per_acre": "₹500–1,100",
        "critical_timing": "At VT/R1 stage (tasseling/silking)",
    },
    "Citrus Greening": {
        "chemicals": [
            {"name": "Imidacloprid 17.8% SL",  "dose": "0.5 ml/L water", "cost_per_litre": "₹1200–1800","applications": "Ongoing"},
            {"name": "Thiamethoxam 25% WG",    "dose": "0.5 g/L water",  "cost_per_kg": "₹2500–3500","applications": "Ongoing"},
        ],
        "organic": "No cure — focus on psyllid control. Yellow sticky traps.",
        "where_to_buy": "Licensed dealers. Contact State Horticulture Department immediately.",
        "cost_estimate_per_acre": "₹2,000–5,000 (ongoing vector control)",
        "critical_timing": "EMERGENCY — report to KVK immediately. Remove infected trees.",
    },
    "Spider Mites": {
        "chemicals": [
            {"name": "Abamectin 1.8% EC",      "dose": "1 ml/L water",   "cost_per_litre": "₹1500–2200","applications": 2},
            {"name": "Spiromesifen 22.9% SC",  "dose": "1 ml/L water",   "cost_per_litre": "₹2000–3000","applications": 2},
            {"name": "Fenazaquin 10% EC",      "dose": "2 ml/L water",   "cost_per_litre": "₹800–1200","applications": 2},
        ],
        "organic": "Neem oil 5ml/L. Predatory mites (Neoseiulus californicus). Strong water jet spray.",
        "where_to_buy": "Agri input shops. Bioagents from NBAII, Bangalore",
        "cost_estimate_per_acre": "₹600–1,200",
        "critical_timing": "Cover leaf undersides thoroughly. Rotate miticides to prevent resistance.",
    },
    "Mosaic Virus": {
        "chemicals": [
            {"name": "No chemical cure — prevention only", "dose": "N/A", "cost_per_kg": "N/A", "applications": 0},
        ],
        "organic": "Remove infected plants. Sanitise tools with bleach (10%). Control aphid vectors.",
        "where_to_buy": "Virus-resistant seeds: national seed companies (MAHYCO, Syngenta)",
        "cost_estimate_per_acre": "₹300–600 (prevention/vector control)",
        "critical_timing": "Remove infected plants immediately to protect healthy ones",
    },
    "Yellow Leaf Curl Virus": {
        "chemicals": [
            {"name": "Imidacloprid 17.8% SL",  "dose": "0.5 ml/L water", "cost_per_litre": "₹1200–1800","applications": "Preventive"},
            {"name": "Thiamethoxam 25% WG",    "dose": "0.5 g/L water",  "cost_per_kg": "₹2500–3500","applications": "Preventive"},
        ],
        "organic": "Yellow sticky traps. Reflective mulch to repel whitefly.",
        "where_to_buy": "Licensed dealers. Traps from agri shops and online",
        "cost_estimate_per_acre": "₹800–1,500 (vector control)",
        "critical_timing": "Remove infected plants immediately. Protect healthy plants with insecticide.",
    },
}

def get_treatment(disease_name: str) -> dict | None:
    """Fuzzy-match disease name to treatment DB."""
    if not disease_name:
        return None
    dn = disease_name.lower()
    for key in TREATMENT_DB:
        if key.lower() in dn or dn in key.lower():
            return TREATMENT_DB[key]
    # Token match
    dn_tokens = set(dn.split())
    best, best_score = None, 0
    for key in TREATMENT_DB:
        score = len(dn_tokens & set(key.lower().split()))
        if score > best_score:
            best_score, best = score, key
    return TREATMENT_DB.get(best) if best_score > 0 else None


# ══════════════════════════════════════════════════════════════════
# 6. TRANSLATIONS — Hindi + regional strings
# ══════════════════════════════════════════════════════════════════
TRANSLATIONS = {
    "en": {
        "upload_prompt":    "Upload a leaf photo or try a sample",
        "analysing":        "Analysing leaf…",
        "confidence":       "Model confidence",
        "recommended_action": "Recommended Action",
        "healthy":          "HEALTHY",
        "moderate":         "MODERATE",
        "high_risk":        "HIGH RISK",
        "critical":         "CRITICAL",
        "scan_history":     "Scan History",
        "field_scan":       "Field Batch Scan",
        "weather_risk":     "Spread Risk Today",
        "treatment_cost":   "Treatment & Cost",
        "save_scan":        "Save to Field Diary",
        "download_report":  "Download Report",
        "spray_today":      "Spray Today",
        "no_spray_needed":  "No immediate spray needed",
        "infection_rate":   "Field Infection Rate",
        "notes_placeholder":"Add notes (optional)…",
        "farmer_name":      "Farmer Name",
        "field_name":       "Field/Plot Name",
        "gradcam_title":    "Why did AI say this?",
        "gradcam_desc":     "Red = regions that triggered the disease prediction",
        "kvk_helpline":     "KVK Helpline: 1800-180-1551 (free, Mon–Sat 6am–10pm)",
    },
    "hi": {
        "upload_prompt":    "पत्ती की फोटो अपलोड करें या नमूना आज़माएं",
        "analysing":        "पत्ती का विश्लेषण हो रहा है…",
        "confidence":       "मॉडल की विश्वसनीयता",
        "recommended_action": "सुझाया गया उपाय",
        "healthy":          "स्वस्थ",
        "moderate":         "मध्यम खतरा",
        "high_risk":        "उच्च खतरा",
        "critical":         "गंभीर",
        "scan_history":     "स्कैन इतिहास",
        "field_scan":       "खेत की बैच स्कैन",
        "weather_risk":     "आज प्रसार का जोखिम",
        "treatment_cost":   "उपचार और लागत",
        "save_scan":        "खेत डायरी में सहेजें",
        "download_report":  "रिपोर्ट डाउनलोड करें",
        "spray_today":      "आज स्प्रे करें",
        "no_spray_needed":  "तुरंत स्प्रे की जरूरत नहीं",
        "infection_rate":   "खेत में संक्रमण दर",
        "notes_placeholder":"नोट्स जोड़ें (वैकल्पिक)…",
        "farmer_name":      "किसान का नाम",
        "field_name":       "खेत/प्लॉट का नाम",
        "gradcam_title":    "AI ने यह क्यों कहा?",
        "gradcam_desc":     "लाल = वे क्षेत्र जिन्होंने बीमारी की पहचान को ट्रिगर किया",
        "kvk_helpline":     "KVK हेल्पलाइन: 1800-180-1551 (निःशुल्क, सोम–शनि 6am–10pm)",
    },
    "te": {
        "upload_prompt":    "ఆకు ఫోటో అప్‌లోడ్ చేయండి లేదా నమూనా ప్రయత్నించండి",
        "analysing":        "ఆకును విశ్లేషిస్తోంది…",
        "confidence":       "మోడల్ నమ్మకం",
        "recommended_action": "సిఫారసు చేయబడిన చర్య",
        "healthy":          "ఆరోగ్యంగా ఉంది",
        "moderate":         "మధ్యస్థ ప్రమాదం",
        "high_risk":        "అధిక ప్రమాదం",
        "critical":         "విమర్శనాత్మక",
        "scan_history":     "స్కాన్ చరిత్ర",
        "field_scan":       "పొల బ్యాచ్ స్కాన్",
        "weather_risk":     "ఈరోజు వ్యాప్తి ప్రమాదం",
        "treatment_cost":   "చికిత్స & ఖర్చు",
        "save_scan":        "క్షేత్ర డైరీకి సేవ్ చేయండి",
        "download_report":  "నివేదికను డౌన్‌లోడ్ చేయండి",
        "spray_today":      "ఈరోజు స్ప్రే చేయండి",
        "no_spray_needed":  "తక్షణ స్ప్రే అవసరం లేదు",
        "infection_rate":   "పొల సంక్రమణ రేటు",
        "notes_placeholder":"గమనికలు జోడించండి (ఐచ్ఛికం)…",
        "farmer_name":      "రైతు పేరు",
        "field_name":       "పొల/ప్లాట్ పేరు",
        "gradcam_title":    "AI ఇది ఎందుకు చెప్పింది?",
        "gradcam_desc":     "ఎరుపు = వ్యాధి అంచనాను ప్రేరేపించిన ప్రాంతాలు",
        "kvk_helpline":     "KVK హెల్ప్‌లైన్: 1800-180-1551 (ఉచిత)",
    },
    "mr": {
        "upload_prompt":    "पानाचा फोटो अपलोड करा किंवा नमुना वापरून पहा",
        "analysing":        "पान विश्लेषण होत आहे…",
        "confidence":       "मॉडेलचा आत्मविश्वास",
        "recommended_action": "शिफारस केलेली कृती",
        "healthy":          "निरोगी",
        "moderate":         "मध्यम धोका",
        "high_risk":        "उच्च धोका",
        "critical":         "गंभीर",
        "scan_history":     "स्कॅन इतिहास",
        "field_scan":       "शेत बॅच स्कॅन",
        "weather_risk":     "आजचा प्रसार धोका",
        "treatment_cost":   "उपचार आणि खर्च",
        "save_scan":        "शेत डायरीत जतन करा",
        "download_report":  "अहवाल डाउनलोड करा",
        "spray_today":      "आज फवारणी करा",
        "no_spray_needed":  "तात्काळ फवारणीची गरज नाही",
        "infection_rate":   "शेतातील संसर्ग दर",
        "notes_placeholder":"नोट्स जोडा (पर्यायी)…",
        "farmer_name":      "शेतकऱ्याचे नाव",
        "field_name":       "शेत/भूखंडाचे नाव",
        "gradcam_title":    "AI ने हे का सांगितले?",
        "gradcam_desc":     "लाल = रोग ओळखण्यास कारणीभूत क्षेत्रे",
        "kvk_helpline":     "KVK हेल्पलाइन: 1800-180-1551 (मोफत)",
    },
}

def t(key: str, lang: str = "en") -> str:
    """Translate a UI string key to the given language."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(
        key, TRANSLATIONS["en"].get(key, key))

LANGUAGE_OPTIONS = {
    "English": "en",
    "हिंदी (Hindi)": "hi",
    "తెలుగు (Telugu)": "te",
    "मराठी (Marathi)": "mr",
}
