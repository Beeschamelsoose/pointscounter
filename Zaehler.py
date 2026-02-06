import sys, time
#print(sys.executable)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, exposure
from skimage.feature import blob_dog, blob_doh
from skimage.io import imread
from skimage.transform import resize
from skimage.util import invert
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
import csv

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"



def blobs_erkennen(img,min_sigma,max_sigma,thresh):
    blobs_dog = blob_dog(
        img,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=thresh
    ) 
    blobs_dog[:, 2] *= np.sqrt(2)
    return blobs_dog


def show_blobs(img, blobs,labels=None, cluster_colours=None, title="Blobs"):
    """
    Zeigt die Blobs an. 
    Wenn labels + cluster_colors gegeben, werden die Blobs in Clusterfarbe angezeigt.
    
    blobs: Array (y,x,r)
    labels: Array der Clusterzuordnung pro Blob
    cluster_colors: Array (n_clusters, 3), Werte 0..255
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, cmap="gray")
    for i,(y, x, r) in enumerate(blobs):
        if labels is not None and cluster_colours is not None:
            colour = cluster_colours[labels[i]]

        else:
            colour = (1,0,0)

        colour = np.array(colour)
        if colour.max() > 1.0:
           
            colour = colour / 255.0
            

        circ = plt.Circle((x, y), r/2, color=colour , fill=True, linewidth=2)
        ax.add_patch(circ)
    ax.set_title(title)
    ax.axis("off")
    return

def show_img(img,title,cmap):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img,cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    return

def get_colour(x,y,img,r=2):
    y = int(round(y))
    x = int(round(x))
    H, W = img.shape[:2]

    y0, y1 = max(0, y-r), min(H, y+r+1)
    x0, x1 = max(0, x-r), min(W, x+r+1)

    patch = img[y0:y1, x0:x1]
    return patch.mean(axis=(0, 1))

def print_colour(text,mono, rgb):
    r,g,b = map(int,rgb)
    print(f"{mono}\033[38;2;{r};{g};{b}m{text}\033[0m")
    return

def find_best_k(colours, k_min=2,k_max=15):
    best_k = k_min
    best_score = -1

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = kmeans.fit_predict(colours)
        score = silhouette_score(colours, labels)
        if args.debug:
            print(f"k={k:2d} | Silhouette-Score = {score:.3f}")

        if score > best_score:
            best_score = score
            best_k = k

    return best_k

def quantize_rgb_image(img, step=8):
    return (img // step) * step

def merge_black(img, thresh=20):
    mask = (img[..., 0] < thresh) & (img[..., 1] < thresh) & (img[..., 2] < thresh)
    img = img.copy()
    img[mask] = (0, 0, 0)
    return img

parser = argparse.ArgumentParser(description="Blob-Erkennung & Farb-Clustering")
parser.add_argument("--img", type=str, default="./img/FR0P1.jpg", help="Pfad zur Bilddatei")
parser.add_argument("--dia",type=int, default=70, help="Ungefaehrer Punkturchmesser")
parser.add_argument("--debug", action="store_true",help="erkannte Punkte im Monochrombild markieren")
parser.add_argument("--scale",type=int, default=4,help="Verkleinerungsfaktor")
parser.add_argument("--csv", action="store_true",help="Erkannte Farben als CSV speichern")
parser.add_argument("--allpts",action="store_true", help="Gibt gesamte erkannte Farbliste aus")

args = parser.parse_args()

img_path= args.img
diameter=args.dia


if args.img is None:
    img_path + str(input("Pfad zum Bild:"))
else:
    img_path is args.img

scale=1/args.scale
v_thresh=254
s_thresh=25

dia_min= diameter*0.8/2
dia_max=diameter*1.1/2
min_sigma = dia_min/(np.sqrt(2)) * scale
max_sigma = dia_max/(np.sqrt(2)) * scale


print("Parameter:")
print("Bild:", img_path)
print("Punktgroesse:",diameter)
print("Verkleinerungsfaktor:",args.scale,"x")
if args.debug:
    print("Debug-Modus aktiv")

    print(f'Minimaler Punktdurchmesser(skaliert):{dia_min}')
    print(f'Maximaler Punktdurchmesser(skaliert):{dia_max}')


img=cv2.imread(img_path)

h,w = img.shape[:2]

img_resized = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
img_cv2_color= cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(img_hsv)

mask = np.zeros_like(v, dtype=np.uint8)

mask[(v >= v_thresh) & (s <= s_thresh)] = 255


#v[v < 254] = 0 

img_cv2 = cv2.bitwise_not(mask)  # Hintergrund schwarz, Blobs hell
img_mono = mask.astype(np.float32)/255.0
img_mono_inv = img_cv2.astype(np.float32)/255.0

img_cv2_color = merge_black(img_cv2_color, thresh=20)
img_cv2_color = quantize_rgb_image(img_cv2_color, step=16)

img_rgb = img_cv2_color.astype(np.float32)/255.0


    
#plt.imshow(img_mono_inv,cmap='gray')

start=time.time_ns()
blobs = blobs_erkennen(img_mono_inv,min_sigma=min_sigma,max_sigma=max_sigma,thresh=0.02)
if args.debug:
    print("DoG time: {:.2f}s".format((time.time_ns()-start)/1e9))

print(f'Anzahl an Erkannten Punkten: {blobs.shape[0]}')

colours= [get_colour(x,y,img_rgb)for (y,x,_) in blobs]

colours = np.array(colours)

best_k = find_best_k(colours, k_min=2, k_max=20)

print("Anzahl an erkannten Farben:", best_k)

kmeans = KMeans(n_clusters=best_k,n_init=10, random_state=0)
labels = kmeans.fit_predict(colours)

clustered_blobs = {i: []for i in range(best_k)}

for (x,y,sigma), label in zip(blobs, labels):
    clustered_blobs[label].append((y,x,sigma))

cluster_colours = kmeans.cluster_centers_
cluster_colours_255 = (cluster_colours*255).round().astype(np.uint8) 

unique, counts = np.unique(labels, return_counts=True)
cluster_counts = dict(zip(unique, counts))

""" print("\nCluster-Overview:")
for i in range(best_k):
    n = cluster_counts.get(i, 0)
    rgb = cluster_colours_255[i]
    txt_mono=f"Cluster {i:2d}: {n:4d} Punkte | RGB = "
    txt = f"{tuple(rgb)}"
    print_colour(txt,txt_mono, rgb) """

print("\nFarbcluster (nach Anzahl sortiert):")

order = sorted(range(best_k), key=lambda i: cluster_counts.get(i, 0), reverse=True)

for i in order:
    n = cluster_counts.get(i, 0)
    rgb = cluster_colours_255[i]
    rgb_int = tuple(int(c)for c in rgb)
    rgb_hex = "#{:02X}{:02X}{:02X}".format(*rgb_int)
    if n == 0:
        txt_mono = f"Cluster {i:2d}: LEER"
        txt = f" (leer) | RGB = {rgb_int} | HEX = {rgb_hex}"
    else:
        txt_mono = f"Cluster {i:2d}: {n:4d} Punkte | RGB = "
        txt = f" {tuple(rgb_int)} | HEX= {rgb_hex}"

    print_colour(txt,txt_mono, rgb)

if args.allpts:
    from collections import Counter
    colours_255 = (colours *255).round().astype(np.uint8)

    rgb_tuples = [tuple(map(int, c)) for c in colours_255]
    counter = Counter(rgb_tuples)

    print("\nExakte RGB-Werte aller erkannten Punkte (ohne Clustering):")
    print("--------------------------------------------------------")

    # nach Häufigkeit sortiert
    for rgb, n in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        rgb_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
        txt_mono = f"{n:4d}x RGB = "
        txt = f"{rgb} | HEX = {rgb_hex}"
        print_colour(txt, txt_mono, rgb)

    print(f"\nUnterschiedliche RGB-Werte: {len(counter)}")
    print(f"Gesamtanzahl erkannter Punkte: {len(colours_255)}")


#cv2.imshow("Blobs", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

#show_img(img_hsv,'IMG_HSV')
if args.debug:
    show_blobs(img_mono_inv,blobs,labels=labels, cluster_colours=cluster_colours,title='Punkterkennung:')
    plt.show()

if args.csv:
    timestr = time.strftime("%d%m%Y-%H%M%S")

    if args.allpts:
        csv_file = f"punkte_rgb_roh-{timestr}.csv"

        from collections import Counter
        counter = Counter(tuple(map(int, c)) for c in colours_255)

        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Anzahl", "R", "G", "B", "HEX"])

            for rgb, n in sorted(counter.items(), key=lambda x: x[1], reverse=True):
                rgb_hex = "#{:02X}{:02X}{:02X}".format(*rgb)
                writer.writerow([n, *rgb, rgb_hex])

        print(f"RGB-Rohdaten wurden nach '{csv_file}' exportiert.")
     
    
    csv_file = f"cluster_ausgabe-{timestr}.csv"

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["ClusterID", "AnzahlPunkte", "R", "G", "B", "HEX"])

        # Daten schreiben
        for i in order:
            n = len(clustered_blobs[i])
            rgb = cluster_colours_255[i]
            rgb_int = tuple(int(c) for c in rgb)
            rgb_hex = "#{:02X}{:02X}{:02X}".format(*rgb_int)
            
            writer.writerow([i, n, *rgb_int, rgb_hex])

    

    print(f"Cluster-Daten wurden nach '{csv_file}' exportiert.")
