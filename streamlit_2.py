#!/usr/bin/env python3
"""
Performance-optimierte Streamlit Web-GUI für Zaehler.py / Blob-Analyzer
- Parallele DoG-Berechnung auf allen CPU-Kernen
- Extrem schnelles OpenCV-Rendering (statt lahmem Matplotlib)
- Farbige Vorschau-Kästchen in den Ergebnistabellen
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.feature import blob_dog
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from joblib import Parallel, delayed
import multiprocessing
import io
import time

# Page config
st.set_page_config(
    page_title="Blob- & Farbzähler (Fast)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== HILFSFUNKTIONEN =====

def quantize_rgb_image(img, step=16):
    return (img // step) * step

def merge_black(img, thresh=20):
    mask = (img[..., 0] < thresh) & (img[..., 1] < thresh) & (img[..., 2] < thresh)
    img_copy = img.copy()
    img_copy[mask] = (0, 0, 0)
    return img_copy

def get_colour_vectorized(blobs, img, r=2):
    """ Vektorisierte Farbentnahme für extrem schnelles Auslesen """
    H, W = img.shape[:2]
    colours = []
    for y, x, _ in blobs:
        y_i, x_i = int(round(y)), int(round(x))
        y0, y1 = max(0, y_i - r), min(H, y_i + r + 1)
        x0, x1 = max(0, x_i - r), min(W, x_i + r + 1)
        colours.append(img[y0:y1, x0:x1].mean(axis=(0, 1)))
    return np.array(colours)

def find_best_k(colours, k_min=2, k_max=20):
    best_k = k_min
    best_score = -1
    
    # Subsampling bei sehr vielen Punkten für schnelles Silhouette-Clustering
    if len(colours) > 1000:
        idx = np.random.choice(len(colours), 1000, replace=False)
        sample_colours = colours[idx]
    else:
        sample_colours = colours

    max_k = min(k_max, len(sample_colours) - 1)
    if max_k <= k_min:
        return k_min

    for k in range(k_min, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=0)
        labels = kmeans.fit_predict(sample_colours)
        score = silhouette_score(sample_colours, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k

def run_blob_dog_single_tile(tile, min_sigma, max_sigma, threshold, y_offset, x_offset):
    """ Führt DoG auf einer Bild-Kachel aus """
    blobs = blob_dog(tile, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    if blobs.shape[0] > 0:
        blobs[:, 0] += y_offset
        blobs[:, 1] += x_offset
    return blobs

def parallel_blob_dog(img, min_sigma, max_sigma, threshold, n_jobs=-1):
    """ Parallelisiert die DoG-Gitter-Berechnung auf alle CPU-Kerne """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    # Falls das Bild sehr klein ist oder nur 1 Core vorhanden ist
    if n_jobs == 1 or img.shape[0] < 200 or img.shape[1] < 200:
        blobs = blob_dog(img, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        if blobs.shape[0] > 0:
            blobs[:, 2] *= np.sqrt(2)
        return blobs

    # Bild in Kacheln zerlegen
    n_splits = int(np.ceil(np.sqrt(n_jobs)))
    h, w = img.shape
    dh = int(np.ceil(h / n_splits))
    dw = int(np.ceil(w / n_splits))

    # Überlappung (Padding) damit Punkte an Schnittkanten nicht verloren gehen
    pad = int(max_sigma * 3)

    tasks = []
    for i in range(n_splits):
        for j in range(n_splits):
            y0, y1 = i * dh, min((i + 1) * dh, h)
            x0, x1 = j * dw, min((j + 1) * dw, w)

            y0_pad, y1_pad = max(0, y0 - pad), min(h, y1 + pad)
            x0_pad, x1_pad = max(0, x0 - pad), min(w, x1 + pad)

            tile = img[y0_pad:y1_pad, x0_pad:x1_pad]
            tasks.append((tile, min_sigma, max_sigma, threshold, y0_pad, x0_pad))

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_blob_dog_single_tile)(t, ms, Ms, th, yo, xo)
        for t, ms, Ms, th, yo, xo in tasks
    )

    all_blobs = [b for b in results if b.shape[0] > 0]
    if not all_blobs:
        return np.empty((0, 3))

    blobs = np.vstack(all_blobs)

    # Duplikate durch Overlap entfernen
    if len(blobs) > 0:
        # Einfache Entfernung zu naher Blobs
        keep = []
        spatial_tree = {}
        for idx, (y, x, r) in enumerate(blobs):
            key = (int(y // (pad/2)), int(x // (pad/2)))
            if key not in spatial_tree:
                spatial_tree[key] = (y, x)
                keep.append(idx)
        blobs = blobs[keep]

    blobs[:, 2] *= np.sqrt(2)
    return blobs

def render_blobs_opencv(img_bg, blobs, labels, cluster_colours_255):
    """ Extrem schnelles Zeichnen der Punkte mit OpenCV """
    # Bild in 3-Kanal BGR/RGB umwandeln
    if len(img_bg.shape) == 2:
        img_out = cv2.cvtColor((img_bg * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        img_out = (img_bg * 255).astype(np.uint8)

    for i, (y, x, r) in enumerate(blobs):
        pt_center = (int(round(x)), int(round(y)))
        radius = max(1, int(round(r / 2)))
        
        if labels is not None and cluster_colours_255 is not None:
            c = cluster_colours_255[labels[i]]
            color_bgr = (int(c[2]), int(c[1]), int(c[0]))  # RGB -> BGR für OpenCV
        else:
            color_bgr = (0, 0, 255)  # Rot

        cv2.circle(img_out, pt_center, radius, color_bgr, -1, lineType=cv2.LINE_AA)
        
    return cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

# ===== MAIN APP =====

st.title("🔬 Blob- & Farbzähler (High-Performance)")

# SIDEBAR
st.sidebar.header("⚙️ Einstellungen")
uploaded_file = st.sidebar.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png", "bmp", "tiff"])

st.sidebar.subheader("Verkleinerung & Multiprocessing")
scale_factor = st.sidebar.slider("Verkleinerungsfaktor (x)", min_value=1, max_value=8, value=4, step=1)
cpu_cores = multiprocessing.cpu_count()
n_jobs = st.sidebar.slider("Parallel-Cores (CPU)", min_value=1, max_value=cpu_cores, value=cpu_cores)

st.sidebar.subheader("Blob-Erkennung")
diameter = st.sidebar.slider("Punkt-Durchmesser (px)", min_value=2, max_value=200, value=70, step=1)
dog_threshold = st.sidebar.slider("DoG-Schwellenwert (thresh)", min_value=0.001, max_value=0.100, value=0.020, step=0.005, format="%.3f")

st.sidebar.subheader("Farbdetection (HSV)")
v_thresh = st.sidebar.slider("Min. Helligkeit (V)", min_value=0, max_value=255, value=255, step=1)
s_thresh = st.sidebar.slider("Max. Sättigung (S)", min_value=0, max_value=255, value=25, step=1)

st.sidebar.subheader("Clustering")
fixed_k = st.sidebar.number_input("Anzahl Farben / K (0 = Auto)", min_value=0, max_value=30, value=0, step=1)
show_all_pts = st.sidebar.checkbox("Alle ungeclusterten RGB-Rohwerte anzeigen", value=False)


if uploaded_file is not None:
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ Bilddatei konnte nicht gelesen werden.")
    else:
        h_orig, w_orig = img.shape[:2]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📸 Original-Bild")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("📊 Bild- & System-Info")
            st.markdown(f"""
            - **Dateiname:** {uploaded_file.name}
            - **Original:** {w_orig} × {h_orig} px
            - **Arbeitsgröße:** {int(w_orig / scale_factor)} × {int(h_orig / scale_factor)} px
            - **Verfügbare CPU-Kerne:** {cpu_cores} (genutzt: {n_jobs})
            """)

        if st.button("▶ ANALYSE STARTEN", key="analyze", use_container_width=True):
            with st.spinner(f"⏳ Analysiere Bild parallel auf {n_jobs} CPU-Kernen..."):
                start_time = time.time()

                # 1. Rescale
                scale = 1.0 / scale_factor
                img_resized = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_NEAREST)

                # 2. HSV & Monochrom Masking
                img_cv2_color = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                _, s_ch, v_ch = cv2.split(img_hsv)

                mask = np.zeros_like(v_ch, dtype=np.uint8)
                mask[(v_ch >= v_thresh) & (s_ch <= s_thresh)] = 255

                img_cv2 = cv2.bitwise_not(mask)
                img_mono_inv = img_cv2.astype(np.float32) / 255.0

                # Color Prep
                img_cv2_color = merge_black(img_cv2_color, thresh=20)
                img_cv2_color = quantize_rgb_image(img_cv2_color, step=16)
                img_rgb = img_cv2_color.astype(np.float32) / 255.0

                # 3. Parallelized Blob Detection
                dia_min = diameter * 0.8 / 2
                dia_max = diameter * 1.1 / 2
                min_sigma = dia_min / (np.sqrt(2)) * scale
                max_sigma = dia_max / (np.sqrt(2)) * scale

                blobs = parallel_blob_dog(
                    img_mono_inv,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    threshold=dog_threshold,
                    n_jobs=n_jobs
                )

                if blobs.shape[0] == 0:
                    st.error("❌ Keine Punkte erkannt! Bitte Parameter anpassen.")
                    st.session_state.pop("analysis_results", None)
                else:
                    # 4. Color Extraction & Clustering
                    colours = get_colour_vectorized(blobs, img_rgb, r=2)

                    if fixed_k > 0:
                        best_k = fixed_k
                    else:
                        best_k = find_best_k(colours, k_min=2, k_max=20)

                    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0)
                    labels = kmeans.fit_predict(colours)

                    cluster_colours = kmeans.cluster_centers_
                    cluster_colours_255 = (cluster_colours * 255).round().astype(np.uint8)

                    elapsed = time.time() - start_time

                    st.session_state["analysis_results"] = {
                        "blobs": blobs,
                        "best_k": best_k,
                        "elapsed": elapsed,
                        "labels": labels,
                        "cluster_colours_255": cluster_colours_255,
                        "colours": colours,
                        "img_mono_inv": img_mono_inv
                    }

        # ERGEBNIS-AUSGABE
        if "analysis_results" in st.session_state:
            res = st.session_state["analysis_results"]
            blobs = res["blobs"]
            best_k = res["best_k"]
            elapsed = res["elapsed"]
            labels = res["labels"]
            cluster_colours_255 = res["cluster_colours_255"]
            colours = res["colours"]
            img_mono_inv = res["img_mono_inv"]

            st.success(f"⚡ Analyse abgeschlossen in nur **{elapsed:.2f} Sekunden**!")

            c1, c2, c3 = st.columns(3)
            c1.metric("🎯 Erkannte Punkte", f"{blobs.shape[0]:,}")
            c2.metric("🎨 Farbklassen", f"{best_k}")
            c3.metric("⏱️ Laufzeit", f"{elapsed:.2f} s")

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.subheader("📍 Schnelle Vorschau (OpenCV)")
                # Blitzschnelles Rendering
                rendered_img = render_blobs_opencv(img_mono_inv, blobs, labels, cluster_colours_255)
                st.image(rendered_img, use_container_width=True)

            with col_right:
                st.subheader("🎨 Farbcluster (mit Farbfeldern)")
                unique, counts = np.unique(labels, return_counts=True)
                cluster_counts = dict(zip(unique, counts))
                order = sorted(range(best_k), key=lambda i: cluster_counts.get(i, 0), reverse=True)

                cluster_data = []
                for i in order:
                    n = cluster_counts.get(i, 0)
                    rgb = cluster_colours_255[i]
                    rgb_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
                    rgb_hex = "#{:02X}{:02X}{:02X}".format(*rgb_tuple)

                    cluster_data.append({
                        "Cluster ID": i,
                        "Anzahl Punkte": n,
                        "Farbe (HEX)": rgb_hex,
                        "RGB Code": f"rgb{rgb_tuple}",
                        "Anteil": f"{(100 * n / blobs.shape[0]):.2f} %"
                    })

                df_clusters = pd.DataFrame(cluster_data)

                # FARBLICHE GESTALTUNG DER TABELLE
                def style_color_column(val):
                    # Hex-Farbwert parsen
                    if isinstance(val, str) and val.startswith("#"):
                        hex_val = val.lstrip("#")
                        r, g, b = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
                        # Textfarbe für guten Kontrast berechnen (Helligkeit YIQ)
                        yiq = ((r * 299) + (g * 587) + (b * 114)) / 1000
                        text_color = "#000000" if yiq >= 128 else "#FFFFFF"
                        return f"background-color: {val}; color: {text_color}; font-weight: bold; text-align: center;"
                    return ""

                styled_df = df_clusters.style.map(style_color_column, subset=["Farbe (HEX)"])
                st.dataframe(styled_df, use_container_width=True)

                # CSV Download (Reiner Text)
                csv_buffer = io.StringIO()
                df_clusters.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="💾 Cluster-Ergebnisse als CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"cluster_ausgabe_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            if show_all_pts:
                st.subheader("📋 Ungeclusterte RGB-Rohdaten")
                colours_255 = (colours * 255).round().astype(np.uint8)
                rgb_tuples = [tuple(map(int, c)) for c in colours_255]
                counter = Counter(rgb_tuples)

                raw_data = []
                for rgb, n in sorted(counter.items(), key=lambda x: x[1], reverse=True):
                    rgb_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
                    raw_data.append({
                        "Anzahl": n,
                        "Farbe (HEX)": rgb_hex,
                        "R": rgb[0], "G": rgb[1], "B": rgb[2]
                    })

                df_raw = pd.DataFrame(raw_data)
                styled_df_raw = df_raw.style.map(style_color_column, subset=["Farbe (HEX)"])
                st.dataframe(styled_df_raw, use_container_width=True)

else:
    st.info("Bitte lade in der linken Seitenleiste ein Bild hoch, um die geglättete Analyse zu starten.")