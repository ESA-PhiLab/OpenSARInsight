import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from dataset_generation_scripts.utils import get_config
cfg = get_config("DGS_CONFIG_PATH")

# ==========================================================
# USER PARAMETERS
# ==========================================================
BASE_PATH = cfg["fd_dataset_generation"]["base_path"]
N_START = cfg["fd_dataset_generation"]["n_start"]
N_END   = cfg["fd_dataset_generation"]["n_end"]

POL     = cfg["fd_dataset_generation"]["pol"]
PREFIX  = cfg["fd_dataset_generation"]["prefix"]

# ==========================================================
# PATH HELPERS
# ==========================================================
def slc_path(n):
    return os.path.join(
        BASE_PATH,
        "cp",
        f"{PREFIX}_{n}_SLC_{POL}.tiff"
    )

def xml_path(n):
    return os.path.join(
        BASE_PATH,
        "label",
        f"{PREFIX}_{n}.xml"
    )

# ==========================================================
# XML → Scene polygons
# ==========================================================
def read_scene_polygons(xml_path, kind):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tag = (
        ".//List_of_FloodEvents//Polygon"
        if kind == "Flood"
        else ".//List_of_WaterBodys//Polygon"
    )

    polys = []

    for poly in root.findall(tag):
        s = poly.find("Scene_Sample")
        l = poly.find("Scene_Line")
        if s is None or l is None or not s.text or not l.text:
            continue

        xs = np.array([float(v) for v in s.text.split()]) - 1
        ys = np.array([float(v) for v in l.text.split()]) - 1

        if xs.size >= 3:
            polys.append((xs, ys))

    return polys

# ==========================================================
# LOAD ALL SLCs AND XMLs
# ==========================================================
slcs   = []
floods = []
waters = []

print("Loading SLCs and XMLs...")

for n in range(N_START, N_END + 1):
    slc_file = slc_path(n)
    xml_file = xml_path(n)

    if not os.path.isfile(slc_file):
        raise FileNotFoundError(f"Missing SLC: {slc_file}")
    if not os.path.isfile(xml_file):
        raise FileNotFoundError(f"Missing XML: {xml_file}")

    slc = tifffile.imread(slc_file)
    mag = np.abs(slc)

    slcs.append(mag)
    floods.append(read_scene_polygons(xml_file, "Flood"))
    waters.append(read_scene_polygons(xml_file, "Water"))

    print(f"  Loaded N={n}")

H, W = slcs[0].shape
print(f"Patch size: {H} x {W}")

# ==========================================================
# PLOT → ALL N FROM LEFT TO RIGHT
# ==========================================================
num = len(slcs)
fig, ax = plt.subplots(1, num, figsize=(4 * num, 6), sharex=True, sharey=True)

if num == 1:
    ax = [ax]

for i, n in enumerate(range(N_START, N_END + 1)):
    ax[i].imshow(20 * np.log10(slcs[i] + 1e-6), cmap="gray", origin="upper")

    for xs, ys in waters[i]:
        ax[i].plot(xs, ys, "b-", linewidth=2)

    for xs, ys in floods[i]:
        ax[i].plot(xs, ys, "r-", linewidth=2)

    ax[i].set_title(f"N = {n}")
    ax[i].set_xlim(0, W)
    ax[i].set_ylim(H, 0)
    ax[i].set_aspect("equal")
    ax[i].set_xlabel("Scene Sample")

ax[0].set_ylabel("Scene Line")

plt.tight_layout()

# Replace plt.show() with plt.savefig to save the plot instead of displaying it
output_path = os.path.join(BASE_PATH, "plot_output.png")
plt.savefig(output_path)
print(f"✅ Plot saved successfully at {output_path}")