"""
example_beampattern.py — Calcula e plota o beampattern de um ULA.

Mostra o uso de steering_vector e beampattern.
Execute a partir da raiz do projeto:  python examples/example_beampattern.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arrays import generate_ula, steering_vector, beampattern

LAM = 1.0
ula = generate_ula(12, LAM / 2)

# vetor diretor para uma direção qualquer
a = steering_vector(ula, azimuth=0.0, elevation=30.0, wavelength=LAM)
print("Vetor diretor (primeiros 3 elementos):", np.round(a[:3], 3))
print("Módulos (devem ser 1):", np.round(np.abs(a[:3]), 3))

# beampattern no corte de elevação
theta = np.linspace(-90, 90, 1801)
Bdb = beampattern(ula, azimuth=0.0, elevation=theta, wavelength=LAM)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(theta, Bdb)
ax.set_xlabel("Elevação θ [°]"); ax.set_ylabel("Ganho normalizado [dB]")
ax.set_title("Beampattern — ULA M=12, d=λ/2")
ax.set_ylim(-40, 2); ax.grid(True, alpha=.3)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figures", "exemplo_beampattern.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120)
print("Figura salva em figures/exemplo_beampattern.png")
