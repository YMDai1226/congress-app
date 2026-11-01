"""
File: Blazealert.py
Author: Yiming Dai,email:DouDouDai316@gmail.com,9th grader san marino high school
Date: 2026-10-24
Description: CAC challenge 
"""


#!/usr/bin/env python3
import os
import sys
import webbrowser
from datetime import date as date_type
from io import StringIO
from pathlib import Path
from html import escape
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

try:
    import folium
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import requests
except ImportError as exc:
    raise SystemExit("Missing package: " + str(exc.name) + "\nInstall: python3 -m pip install pandas requests numpy matplotlib folium") from None

CITIES = {
    "los angeles": {
        "bbox": (-118.95, 33.60, -117.60, 34.85),
        "lat": 34.1478,
        "lon": -118.1445
    },
    "san francisco": {
        "bbox": (-122.55, 37.70, -122.35, 37.85),
        "lat": 37.7749,
        "lon": -122.4194
    },
    "san diego": {
        "bbox": (-117.30, 32.50, -116.70, 33.10),
        "lat": 32.7157,
        "lon": -117.1611
    },
    "sacramento": {
        "bbox": (-121.60, 38.35, -121.30, 38.75),
        "lat": 38.5816,
        "lon": -121.4944
    },
    "fresno": {
        "bbox": (-120.00, 36.60, -119.60, 37.00),
        "lat": 36.7468,
        "lon": -119.7726
    },
    "santa barbara": {
        "bbox": (-120.00, 34.35, -119.50, 34.55),
        "lat": 34.4208,
        "lon": -119.6982
    }
}

DEFAULT_MAP_KEY = "e84844296d86ffba3aedce06b0023f20"

def get_firms_data(map_key, bbox, source="VIIRS_NOAA21_NRT", days=1):
    if not 1 <= int(days) <= 5:
        raise ValueError("Days must be between 1 and 5.")
    west, south, east, north = bbox
    area = f"{west},{south},{east},{north}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{area}/{int(days)}"
    try:
        response = requests.get(url, timeout=45)
    except requests.RequestException:
        raise RuntimeError("FIRMS request failed. Check your network connection.") from None
    if not response.ok:
        raise RuntimeError(f"FIRMS HTTP {response.status_code}. Verify API access.")
    text = response.text.strip()
    if not text:
        return pd.DataFrame()
    if text.lower().startswith(("error", "invalid", "map key")):
        raise RuntimeError("FIRMS returned an error: " + text[:300].replace(map_key, "[REDACTED]"))
    return pd.read_csv(StringIO(text))

def prepare_firms_data(df):
    result = df.copy()
    if result.empty:
        return result
    for col in ["latitude", "longitude", "frp"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    time_text = result["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    result["timestamp_utc"] = pd.to_datetime(result["acq_date"].astype(str) + " " + time_text, format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    result["timestamp_california"] = result["timestamp_utc"].dt.tz_convert("America/Los_Angeles")
    return result

def haversine_miles(lat1, lon1, lat2, lon2):
    r_km = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    km = 2 * r_km * np.arcsin(np.sqrt(a))
    return km / 1.609344

def confidence_color(val):
    return {"h": "#8b0000", "n": "#ff8c00", "l": "#ffd700"}.get(str(val).lower(), "#ff0000")

def build_fire_map(df, station_name, lat, lon, radius_miles=30):
    m = folium.Map(location=[lat, lon], zoom_start=8, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Esri World Imagery", overlay=False, control=True, max_zoom=19, show=True
    ).add_to(m)
    
    title_html = '''
    <div style="position: fixed; top: 15px; left: 60px; z-index:9999; font-size:16px; background-color: rgba(15, 23, 42, 0.9); color: #f8fafc; padding: 10px 20px; border-radius: 8px; border: 1px solid #334155; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        BlazeAlert
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    folium.Marker([lat, lon], tooltip=station_name, popup=f"<b>{escape(station_name)}</b>", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
    folium.Circle([lat, lon], radius=radius_miles * 1609.344, color="#2563eb", weight=2, fill=False).add_to(m)
    
    for _, row in df.iterrows():
        conf = row.get("confidence", "unknown")
        ts = row.get("timestamp_california", "unknown")
        frp = row.get("frp", np.nan)
        dist = row.get("distance_miles", np.nan)
        sat = row.get("satellite", "unknown")
        popup = folium.Popup(f"<b>BlazeAlert Detection</b><br>Satellite: {sat}<br>Time: {ts}<br>Confidence: {conf}<br>FRP: {frp:.1f} MW<br>Distance: {dist:.1f} mi", max_width=360)
        radius = float(np.clip(4 + np.sqrt(max(float(frp), 0)), 4, 14))
        folium.CircleMarker([row["latitude"], row["longitude"]], radius=radius, color=confidence_color(conf), weight=1, fill=True, fill_opacity=0.75, popup=popup).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

FORM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BlazeAlert</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #f97316;
            --accent-hover: #ea580c;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            box-sizing: border-box;
        }
        .container {
            width: 100%;
            max-width: 520px;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            font-size: 2.2rem;
            color: var(--accent);
            margin: 0 0 8px 0;
            letter-spacing: -0.025em;
        }
        p.subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin: 0;
        }
        label {
            display: block;
            margin-top: 18px;
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-main);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        input, select {
            width: 100%;
            padding: 12px 16px;
            margin-top: 6px;
            background-color: #0f172a;
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 1rem;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.2);
        }
        button {
            background-color: var(--accent);
            color: white;
            border: none;
            padding: 14px 20px;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 32px;
            width: 100%;
            transition: background-color 0.2s, transform 0.1s;
        }
        button:hover {
            background-color: var(--accent-hover);
        }
        button:active {
            transform: scale(0.98);
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>BlazeAlert</h1>
        <p class="subtitle">Real-Time Wildfire Detection & Intelligence</p>
    </div>
    <form method="POST" action="/">
        <label>City Name</label>
        <input type="text" name="city_name" required placeholder="e.g., Los Angeles, San Francisco">

        <label>Search Days (1 to 5)</label>
        <select name="days">
            <option value="1">1 Day</option>
            <option value="2">2 Days</option>
            <option value="3">3 Days</option>
            <option value="4">4 Days</option>
            <option value="5" selected>5 Days</option>
        </select>

        <label>Alert Radius (miles)</label>
        <input type="number" step="any" name="radius_miles" value="30" required>

        <button type="submit">Launch BlazeAlert Map</button>
    </form>
</div>
</body>
</html>
"""

class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(FORM_HTML.encode("utf-8"))

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(length).decode('utf-8')
        params = parse_qs(data)
        try:
            city_input = params.get("city_name", ["los angeles"])[0].strip().lower()
            days = int(params.get("days", [5])[0])
            radius = float(params.get("radius_miles", [30])[0])

            city_data = CITIES.get(city_input, CITIES["los angeles"])
            bbox = city_data["bbox"]
            lat = city_data["lat"]
            lon = city_data["lon"]
            station_name = f"{city_input.title()} Station"

            frame = get_firms_data(DEFAULT_MAP_KEY, bbox, source="VIIRS_NOAA21_NRT", days=days)
            frame["source_product"] = "VIIRS_NOAA21_NRT"
            fires = prepare_firms_data(frame)

            if fires.empty:
                fires = pd.DataFrame(columns=["latitude", "longitude", "frp", "confidence", "distance_miles", "source_product"])
            else:
                fires = fires.dropna(subset=["latitude", "longitude"]).copy()
                if "confidence" in fires:
                    fires = fires[fires["confidence"].astype(str).str.lower().isin(["n", "h"])].copy()
                fires["distance_miles"] = haversine_miles(lat, lon, fires["latitude"], fires["longitude"])

            fire_map = build_fire_map(fires, station_name, lat, lon, radius)
            map_html_str = fire_map._repr_html_()

            res = f"""<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>BlazeAlert</title>
                <style>
                    body {{ margin: 0; padding: 0; height: 100vh; overflow: hidden; font-family: system-ui, sans-serif; }}
                    .top-bar {{ position: absolute; top: 0; left: 0; right: 0; height: 50px; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; z-index: 1000; border-bottom: 1px solid #334155; }}
                    .top-bar h2 {{ margin: 0; font-size: 1.2rem; color: #f97316; }}
                    .top-bar a {{ color: #94a3b8; text-decoration: none; font-size: 0.9rem; font-weight: 600; }}
                    .top-bar a:hover {{ color: #f8fafc; }}
                    .map-container {{ position: absolute; top: 50px; bottom: 0; left: 0; right: 0; }}
                    .map-container iframe {{ width: 100%; height: 100%; border: none; }}
                </style>
            </head>
            <body>
                <div class="top-bar">
                    <h2>BlazeAlert — {city_input.title()} ({len(fires):,} detections)</h2>
                    <a href="/">← New Search</a>
                </div>
                <div class="map-container">
                    {map_html_str}
                </div>
            </body>
            </html>"""
            
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(res.encode("utf-8"))
        except Exception as exc:
            err = f"""<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>BlazeAlert - Error</title>
                <style>
                    body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; padding: 60px 20px; text-align: center; }}
                    .card {{ max-width: 500px; background: #1e293b; padding: 40px; border-radius: 16px; margin: auto; border: 1px solid #334155; }}
                    h2 {{ color: #ef4444; margin-top: 0; }}
                    a {{ color: #f97316; text-decoration: none; font-weight: 600; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <h2>Error Occurred</h2>
                    <p>{escape(str(exc))}</p>
                    <p style="margin-top: 25px;"><a href="/">← Back to Dashboard</a></p>
                </div>
            </body>
            </html>"""
            self.send_response(500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(err.encode("utf-8"))

def main():
    server = HTTPServer(('', 8080), WebHandler)
    webbrowser.open("http://localhost:8080")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()