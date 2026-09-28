#!/usr/bin/env python3
"""
Streamlit Interface basierend auf der Zaehler.py Logik
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import blob_dog
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import io
import time

# Page config
st.set_page_config(
    page_title="Blob- & Farb-Zähler",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== HILFSFUNKTIONEN AUS ZAEHLER.PY =====

def quantize_rgb_image(img, step=16):
    return (img // step) * step

def merge_black(img, thresh=20):
    mask = (img[..., 0] < thresh) & (img[..., 1] < thresh) & (img[..., 2] < thresh)
    img_copy = img.copy()
    img_copy[mask] = (0, 0, 0)
    return img_copy

def get_colour(x, y, img, r=2):
    y = int(round(y))
    x = int(round(x))
    H, W = img.shape[:2]

    y0, y1 = max(0, y-r), min(H, y+r+1)
    x0, x1 = max(0, x-r), min(W, x+r+1)

    patch = img[y0:y1, x0:x1]
    return patch.mean(axis=(0, 1))

def find_best_k(colours, k_min=2, k_max=20):
    best_k = k_min
    best_score = -1

    # Falls nicht genug Farbpunkte existieren
    max_k = min(k_max, len(colours) - 1)
    if max_k <= k_min:
        return k_min

    for k in range(k_min, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = kmeans.fit_predict(colours)
        score = silhouette_score(colours, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k

def plot_blobs_matplotlib(img_mono_inv, blobs, labels, cluster_colours):
    """ Erstellt den Matplotlib Plot analog zu show_blobs aus Zaehler.py """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_mono_inv, cmap="gray")
    
    for i, (y, x, r) in enumerate(blobs):
        if labels is not None and cluster_colours is not None:
            colour = cluster_colours[labels[i]]
        else:
            colour = (1.0, 0.0, 0.0)

        # Sicherstellen, dass die Farbwerte im Bereich 0.0 bis 1.0 liegen
        colour = np.array(colour, dtype=np.float64)
        if colour.max() > 1.0:
            colour = colour / 255.0

        # Wichtig: Werte auf den Bereich [0.0, 1.0] begrenzen
        colour = np.clip(colour, 0.0, 1.0)

        circ = plt.Circle((x, y), r / 2, color=colour, fill=True, linewidth=1.5)
        ax.add_patch(circ)
        
    ax.set_title(f"Erkannte Punkte ({len(blobs)})")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ===== TITLE =====
st.title("🔬 Blob- & Farbzähler")
st.markdown("Automatische Punkterkennung und Farb-Clustering basierend auf `Zaehler.py`")

# ===== SIDEBAR: PARAMETER =====
st.sidebar.header("⚙️ Einstellungen")

uploaded_file = st.sidebar.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png", "bmp", "tiff"])

st.sidebar.subheader("Verkleinerung & Bild")
scale_factor = st.sidebar.slider("Verkleinerungsfaktor (x)", min_value=1, max_value=8, value=4, step=1)

st.sidebar.subheader("Blob-Erkennung")
diameter = st.sidebar.slider("Punkt-Durchmesser (px)", min_value=2, max_value=200, value=70, step=1)
dog_threshold = st.sidebar.slider("DoG-Schwellenwert (thresh)", min_value=0.001, max_value=0.100, value=0.020, step=0.005, format="%.3f")

st.sidebar.subheader("Farbdetection (HSV)")
v_thresh = st.sidebar.slider("Min. Helligkeit (V)", min_value=0, max_value=255, value=254, step=1)
s_thresh = st.sidebar.slider("Max. Sättigung (S)", min_value=0, max_value=255, value=25, step=1)

st.sidebar.subheader("Clustering")
fixed_k = st.sidebar.number_input("Anzahl Farben / K (0 = Auto)", min_value=0, max_value=30, value=0, step=1)
show_all_pts = st.sidebar.checkbox("Alle nicht-geclusterten RGB-Rohwerte anzeigen", value=False)

# ===== HAUPTBEREICH =====
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
            st.subheader("📊 Bild- & Parameterinfo")
            st.markdown(f"""
            - **Dateiname:** {uploaded_file.name}
            - **Auflösung:** {w_orig} × {h_orig} px
            - **Skaliert:** {int(w_orig / scale_factor)} × {int(h_orig / scale_factor)} px
            - **Ungefährer Punkt-Durchmesser:** {diameter} px
            """)
        
        if st.button("▶ ANALYSE STARTEN", key="analyze", use_container_width=True):
            with st.spinner("⏳ Analysiere Bild..."):
                start_time = time.time()
                
                # 1. Skalierung
                scale = 1.0 / scale_factor
                img_resized = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_NEAREST)
                
                # 2. Farbkonvertierungen & Vorverarbeitung aus Zaehler.py
                img_cv2_color = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                
                h_ch, s_ch, v_ch = cv2.split(img_hsv)
                
                mask = np.zeros_like(v_ch, dtype=np.uint8)
                mask[(v_ch >= v_thresh) & (s_ch <= s_thresh)] = 255
                
                img_cv2 = cv2.bitwise_not(mask)
                img_mono_inv = img_cv2.astype(np.float32) / 255.0
                
                # Farbaufbereitung (Quantisierung + Merge Black)
                img_cv2_color = merge_black(img_cv2_color, thresh=20)
                img_cv2_color = quantize_rgb_image(img_cv2_color, step=16)
                img_rgb = img_cv2_color.astype(np.float32) / 255.0
                
                # 3. Blob-Berechnung
                dia_min = diameter * 0.8 / 2
                dia_max = diameter * 1.1 / 2
                min_sigma = dia_min / (np.sqrt(2)) * scale
                max_sigma = dia_max / (np.sqrt(2)) * scale
                
                blobs = blob_dog(
                    img_mono_inv,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    threshold=dog_threshold
                )
                
                if blobs.shape[0] == 0:
                    st.error("❌ Keine Punkte erkannt! Passe den Punkt-Durchmesser, HSV-Schwellenwerte oder DoG-Threshold an.")
                    st.session_state.pop("analysis_results", None)
                else:
                    blobs[:, 2] *= np.sqrt(2)
                    
                    # 4. Farbentnahme pro Punkt
                    colours = np.array([get_colour(x, y, img_rgb) for (y, x, _) in blobs])
                    
                    # 5. Clustering
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
                        "cluster_colours": cluster_colours,
                        "cluster_colours_255": cluster_colours_255,
                        "colours": colours,
                        "img_mono_inv": img_mono_inv
                    }

        # Ergbenisse anzeigen
        if "analysis_results" in st.session_state:
            res = st.session_state["analysis_results"]
            blobs = res["blobs"]
            best_k = res["best_k"]
            elapsed = res["elapsed"]
            labels = res["labels"]
            cluster_colours = res["cluster_colours"]
            cluster_colours_255 = res["cluster_colours_255"]
            colours = res["colours"]
            img_mono_inv = res["img_mono_inv"]

            st.success(f"✓ Analyse erfolgreich durchgeführt in {elapsed:.2f} Sekunden.")
            
            # Kennzahlen
            c1, c2, c3 = st.columns(3)
            c1.metric("🎯 Erkannte Punkte", f"{blobs.shape[0]:,}")
            c2.metric("🎨 Cluster / Farben", f"{best_k}")
            c3.metric("⏱️ Laufzeit", f"{elapsed:.2f} s")
            
            # Visualisierung & Cluster
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.subheader("📍 Visualisierung (Blobs & Cluster-Farben)")
                fig = plot_blobs_matplotlib(img_mono_inv, blobs, labels, cluster_colours)
                st.pyplot(fig)
            
            with col_right:
                st.subheader("🎨 Farbcluster (nach Anzahl sortiert)")
                unique, counts = np.unique(labels, return_counts=True)
                cluster_counts = dict(zip(unique, counts))
                order = sorted(range(best_k), key=lambda i: cluster_counts.get(i, 0), reverse=True)
                
                cluster_data = []
                for i in order:
                    n = cluster_counts.get(i, 0)
                    rgb = cluster_colours_255[i]
                    rgb_int = tuple(int(c) for c in rgb)
                    rgb_hex = "#{:02X}{:02X}{:02X}".format(*rgb_int)
                    
                    cluster_data.append({
                        "Cluster ID": i,
                        "Anzahl Punkte": n,
                        "RGB": str(rgb_int),
                        "HEX": rgb_hex,
                        "Anteil": f"{(100 * n / blobs.shape[0]):.2f} %"
                    })
                
                df_clusters = pd.DataFrame(cluster_data)
                st.dataframe(df_clusters, use_container_width=True)
                
                # CSV Export
                csv_buffer = io.StringIO()
                df_clusters.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="💾 Cluster-Ergebnisse als CSV herunterladen",
                    data=csv_buffer.getvalue(),
                    file_name=f"cluster_ausgabe_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # Optionale Ausgabe aller Rohfarben
            if show_all_pts:
                st.subheader("📋 Exakte RGB-Werte aller erkannten Punkte")
                colours_255 = (colours * 255).round().astype(np.uint8)
                rgb_tuples = [tuple(map(int, c)) for c in colours_255]
                counter = Counter(rgb_tuples)
                
                raw_color_data = []
                for rgb, n in sorted(counter.items(), key=lambda x: x[1], reverse=True):
                    rgb_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
                    raw_color_data.append({
                        "Anzahl": n,
                        "R": rgb[0],
                        "G": rgb[1],
                        "B": rgb[2],
                        "HEX": rgb_hex
                    })
                
                df_raw = pd.DataFrame(raw_color_data)
                st.dataframe(df_raw, use_container_width=True)
                
                csv_raw_buffer = io.StringIO()
                df_raw.to_csv(csv_raw_buffer, index=False)
                st.download_button(
                    label="💾 Rohdaten-RGB als CSV herunterladen",
                    data=csv_raw_buffer.getvalue(),
                    file_name=f"punkte_rgb_roh_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

else:
    st.info("Bitte lade in der linken Seitenleiste ein Bild hoch, um die Analyse zu starten.")