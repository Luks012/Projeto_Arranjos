"""
section7_extra.py
=================
Seção 7 — Múltiplos Sinais (Extra, ampliado).

Investiga como a capacidade de separar duas fontes simultâneas depende de:
  (1) número de sensores N do arranjo receptor;
  (2) espaçamento d entre elementos.

Ferramenta: varredura do beamformer Delay-and-Sum sobre o ângulo, com duas
fontes incoerentes (frequências distintas) chegando de direções diferentes.
A "resolução" é avaliada pela presença de dois lóbulos distintos e pela
profundidade do vale (dip) entre eles.

Insight central sobre d:
  - aumentar d alarga a abertura -> estreita o feixe -> melhora a resolução;
  - porém d > λ/2 provoca lóbulos de difração (grating lobes), i.e.,
    aliasing espacial: direções falsas com ganho igual ao do lóbulo
    principal, gerando ambiguidade.
"""

import numpy as np
import os
FIGDIR = os.environ.get("FIGDIR") or os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arrays import generate_ula, steering_vector, beamformer

C, FC = 3e8, 1e9
LAM = C / FC                      # 0.3 m
FS, DUR = 16000.0, 0.05
t = np.arange(0, DUR, 1 / FS)
s1 = np.sin(2 * np.pi * 500 * t)         # fonte 1
s2 = np.sin(2 * np.pi * 1500 * t)        # fonte 2 (incoerente com a 1)


def scan_power(N, d, th1, th2, scan):
    """Potência na saída do beamformer ao varrer 'scan', com 2 fontes."""
    rx = generate_ula(N, d)
    a1 = steering_vector(rx, 0.0, th1, LAM)
    a2 = steering_vector(rx, 0.0, th2, LAM)
    x = np.outer(a1, s1) + np.outer(a2, s2)
    P = np.array([np.mean(np.abs(beamformer(x, rx, (0.0, a), LAM)) ** 2)
                  for a in scan])
    return P / P.max()


def dip_depth_dB(scan, P_dB, th1, th2):
    """Profundidade do vale entre as duas fontes (dB abaixo do menor pico)."""
    i1 = np.argmin(np.abs(scan - th1))
    i2 = np.argmin(np.abs(scan - th2))
    lo, hi = sorted((i1, i2))
    valley = P_dB[lo:hi + 1].min()
    peak = min(P_dB[i1], P_dB[i2])
    return peak - valley            # quanto maior, mais resolvidas


# =====================================================================
# (1) Influência do NÚMERO DE SENSORES  (d = λ/2 fixo)
# =====================================================================
scan = np.linspace(-40, 40, 1601)
th1, th2 = -6.0, 6.0               # separação fixa de 12°
fig, ax = plt.subplots(figsize=(9, 4.4))
print("=== Influência de N (d=λ/2, fontes em -6° e +6°, Δθ=12°) ===")
for N in (4, 8, 16):
    P = scan_power(N, LAM / 2, th1, th2, scan)
    PdB = 10 * np.log10(P)
    dip = dip_depth_dB(scan, PdB, th1, th2)
    ax.plot(scan, PdB, label=f"N = {N}  (vale: {dip:.1f} dB)")
    print(f"  N={N:2d}: profundidade do vale = {dip:5.2f} dB "
          f"({'resolvido' if dip > 1 else 'NÃO resolvido'})")
for thx in (th1, th2):
    ax.axvline(thx, color="k", ls=":", alpha=.4)
ax.set_xlabel("direção de varredura θ [°]"); ax.set_ylabel("potência [dB]")
ax.set_title("Influência do número de sensores na separação (d = λ/2, Δθ = 12°)")
ax.set_ylim(-30, 2); ax.grid(True, alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig7e_num_sensores.png"), dpi=130); plt.close(fig)


# =====================================================================
# (2) Influência do ESPAÇAMENTO  (N = 8 fixo)
# =====================================================================
scan = np.linspace(-90, 90, 3601)
th1, th2 = -10.0, 10.0            # separação de 20°
fig, ax = plt.subplots(figsize=(9, 4.6))
print("\n=== Influência de d (N=8, fontes em -10° e +10°, Δθ=20°) ===")
for d, lbl in [(LAM / 4, "d = λ/4"), (LAM / 2, "d = λ/2"), (LAM, "d = λ")]:
    P = scan_power(8, d, th1, th2, scan)
    PdB = 10 * np.log10(P)
    dip = dip_depth_dB(scan, PdB, th1, th2)
    ax.plot(scan, PdB, label=f"{lbl}  (vale: {dip:.1f} dB)")
    print(f"  {lbl}: profundidade do vale = {dip:5.2f} dB")
for thx in (th1, th2):
    ax.axvline(thx, color="k", ls=":", alpha=.4)
ax.set_xlabel("direção de varredura θ [°]"); ax.set_ylabel("potência [dB]")
ax.set_title("Influência do espaçamento na separação (N = 8, Δθ = 20°)")
ax.set_ylim(-30, 2); ax.grid(True, alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig7e_espacamento.png"), dpi=130); plt.close(fig)


# =====================================================================
# (3) Aliasing espacial: lóbulo de difração com d > λ/2 (1 fonte)
# =====================================================================
scan = np.linspace(-90, 90, 3601)
th0 = 20.0                        # uma única fonte em 20°
fig, ax = plt.subplots(figsize=(9, 4.4))
print("\n=== Aliasing espacial: 1 fonte em 20° ===")
for d, lbl in [(LAM / 2, "d = λ/2 (sem ambiguidade)"),
               (LAM, "d = λ (lóbulo de difração)")]:
    rx = generate_ula(8, d)
    a0 = steering_vector(rx, 0.0, th0, LAM)
    x = np.outer(a0, s1)
    P = np.array([np.mean(np.abs(beamformer(x, rx, (0.0, a), LAM)) ** 2)
                  for a in scan])
    PdB = 10 * np.log10(P / P.max())
    ax.plot(scan, PdB, label=lbl)
    # detecta picos próximos de 0 dB (lóbulos de igual ganho)
    peaks = scan[(PdB > -0.5)]
    print(f"  {lbl}: direções com ganho ≈ máximo -> {np.round(peaks[::max(1,len(peaks)//5)],0)}")
ax.axvline(th0, color="k", ls=":", alpha=.5, label="fonte real (20°)")
ax.set_xlabel("direção de varredura θ [°]"); ax.set_ylabel("potência [dB]")
ax.set_title("Aliasing espacial: lóbulo de difração para d > λ/2 (N = 8)")
ax.set_ylim(-30, 2); ax.grid(True, alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig7e_aliasing.png"), dpi=130); plt.close(fig)

print("\nFiguras do extra geradas.")
