"""
example_geometrias.py — Demonstra as funções geradoras de geometria.

Cria cada arranjo, imprime as coordenadas e salva uma figura das posições.
Execute a partir da raiz do projeto:  python examples/example_geometrias.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arrays import generate_ula, generate_uca, generate_upa, generate_ucya

LAM = 1.0
arranjos = {
    "ULA": generate_ula(9, LAM / 2),
    "UCA": generate_uca(16, LAM),
    "UPA": generate_upa(4, 4, LAM / 2, LAM / 2),
    "Cilíndrico": generate_ucya(16, 4, LAM, LAM / 2),
}

for nome, pos in arranjos.items():
    print(f"{nome}: {pos.shape[0]} elementos")
    print(np.round(pos / LAM, 3)[:3], "...\n")

fig = plt.figure(figsize=(10, 8))
for i, (nome, pos) in enumerate(arranjos.items(), 1):
    ax = fig.add_subplot(2, 2, i, projection="3d")
    ax.scatter(pos[:, 0] / LAM, pos[:, 1] / LAM, pos[:, 2] / LAM, s=35)
    ax.set_title(nome); ax.set_xlabel("x/λ"); ax.set_ylabel("y/λ"); ax.set_zlabel("z/λ")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figures", "exemplo_geometrias.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120)
print("Figura salva em figures/exemplo_geometrias.png")
