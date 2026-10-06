import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Plot the slipping-cube displacement history.")
parser.add_argument("--input-dir", default=str(CASE_DIR.parent / "OutputData"))
parser.add_argument("--output-file", default=str(CASE_DIR / "Block_miu0.8_a30_NoGauss_mo.png"))
parser.add_argument("--start-frame", type=int, default=0)
parser.add_argument("--end-frame", type=int, default=60, help="Exclusive end frame.")
arguments = parser.parse_args()

params = {
    "backend": "ps",
    "font.size": 26,
    "lines.linewidth": 4.5,
    "lines.markersize": 10,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "xtick.major.pad": 12,
    "ytick.major.pad": 12,
    "axes.labelpad": 8,
    "legend.fontsize": 26,
    "figure.figsize": [12, 9],
    "font.family": "serif",
    "text.usetex": True,
    "font.serif": "Arial",
    "savefig.dpi": 300,
}
rcParams.update(params)

color = [
    (0 / 255, 0 / 255, 0 / 255),
    (255 / 255, 0 / 255, 0 / 255),
    (94 / 255, 114 / 255, 255 / 255),
    (0 / 255, 128 / 255, 0 / 255),
]

q0 = 1000
start = arguments.start_frame
end = arguments.end_frame
vel0 = 0.02
b = 1.0


def calculate(path):
    contact_stress = []
    epslion = []
    time = []
    xpos0 = 0.0
    for printNum in range(start, end):
        grid = np.load(path / "grids" / "MPMGrid{0:06d}.npz".format(printNum))
        particle = np.load(path / "particles" / "MPMParticle{0:06d}.npz".format(printNum))
        contact_stress.append(np.sum(grid["contact_force"][:, 1][:, 1]) / 0.5 / q0)

        if printNum == start:
            xpos0 = np.mean(particle["position"][particle["bodyID"] == 0][:, 0])
        epslion.append((np.mean(particle["position"][particle["bodyID"] == 0][:, 0]) - xpos0))
        time.append(grid["t_current"])
        print(epslion)

    return contact_stress, epslion, time


contact_stress3, epslion3, time3 = calculate(Path(arguments.input_dir).expanduser().resolve())

x1 = np.arange(0.0, 6.0, 0.5)
y1 = 0.5 * (0.5 - 0.2 * 0.866025) * 9.8 * x1 * x1

plt.plot(x1, y1, color=color[3], linestyle="-.", label="Analytical")
plt.plot(time3, epslion3, c="grey", label="MPM (GeoTaichi)")

plt.xlabel("Time, $s$")
plt.ylabel("Displacement, $m$")
plt.xlim([0, 4])
plt.ylim([0, 18])
plt.tight_layout()
plt.legend(frameon=False)
output_file = Path(arguments.output_file).expanduser().resolve()
output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file)
