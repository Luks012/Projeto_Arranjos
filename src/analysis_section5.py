"""
section5.py
===========
Análise da Seção 5: para cada geometria, com PESOS UNIFORMES (Eq. 9):
  - figura de posições 3D
  - beampattern (corte 1D, mapa 2D ou superfície 3D, conforme a geometria)
  - métricas: largura de feixe de meia potência (HPBW, -3 dB),
    nível do lóbulo secundário (SLL), diretividade numérica
  - tabela comparativa (estilo Tabela 1)

Convenção e funções: ver arrays.py. Todos os ângulos em graus.
"""

import numpy as np
import os
FIGDIR = os.environ.get("FIGDIR") or os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arrays import (generate_ula, generate_uca, generate_upa, generate_ucya,
                    beampattern, array_factor)

LAM = 1.0


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------
def hpbw_from_cut(angles_deg, B_dB):
    """Largura de feixe de meia potência (-3 dB) ao redor do pico (corte 1D)."""
    i_pk = np.argmax(B_dB)
    # caminha p/ direita até cruzar -3 dB
    def cross(direction):
        i = i_pk
        while 0 <= i < len(B_dB) and B_dB[i] > -3.0:
            i += direction
        if i < 0 or i >= len(B_dB):
            return None
        # interpolação linear no cruzamento
        a0, a1 = angles_deg[i - direction], angles_deg[i]
        b0, b1 = B_dB[i - direction], B_dB[i]
        return a0 + (-3.0 - b0) * (a1 - a0) / (b1 - b0)
    left = cross(-1)
    right = cross(+1)
    if left is None or right is None:
        return np.nan
    return abs(right - left)


def sll_from_cut(angles_deg, B_dB):
    """Nível do maior lóbulo secundário [dB] (corte 1D)."""
    # detecta máximos locais
    peaks = []
    for i in range(1, len(B_dB) - 1):
        if B_dB[i] > B_dB[i - 1] and B_dB[i] > B_dB[i + 1]:
            peaks.append(B_dB[i])
    peaks = sorted(peaks, reverse=True)
    # o 1º é o lóbulo principal (~0 dB); o 2º é o maior secundário
    return peaks[1] if len(peaks) > 1 else np.nan


def directivity(positions, weights=None, n_phi=361, n_theta=181):
    """
    Diretividade numérica:
        D = max|AF|^2 / <|AF|^2>_esfera
    com elemento de ângulo sólido dΩ = cosθ dθ dφ (θ = elevação).
    """
    phi = np.linspace(0, 360, n_phi)
    th = np.linspace(-90, 90, n_theta)
    PHI, TH = np.meshgrid(phi, th, indexing="xy")
    AF = array_factor(positions, PHI, TH, LAM, weights)
    P = np.abs(AF) ** 2
    w_solid = np.cos(np.deg2rad(TH))
    num = P.max()
    # média ponderada pelo ângulo sólido
    den = np.sum(P * w_solid) / np.sum(w_solid)
    return num / den


# ---------------------------------------------------------------------
# Figuras por geometria
# ---------------------------------------------------------------------
def fig_positions(positions, title, fname):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0] / LAM, positions[:, 1] / LAM, positions[:, 2] / LAM,
               c="tab:blue", s=40)
    ax.set_xlabel("x/λ"); ax.set_ylabel("y/λ"); ax.set_zlabel("z/λ")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, fname), dpi=130); plt.close(fig)


def analyze_ula():
    th = np.linspace(-90, 90, 4001)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows = []
    for M in (9, 16):
        pos = generate_ula(M, LAM / 2)
        B = beampattern(pos, 0.0, th, LAM)               # φ=0, varre θ
        ax.plot(th, B, label=f"M = {M}")
        rows.append(("ULA", M, hpbw_from_cut(th, B), sll_from_cut(th, B),
                     directivity(pos)))
    ax.set_xlabel("Elevação θ [°]"); ax.set_ylabel("Ganho normalizado [dB]")
    ax.set_title("Beampattern ULA (d=λ/2, φ=0°)")
    ax.set_ylim(-40, 2); ax.grid(True, alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig5_ula_bp.png"), dpi=130); plt.close(fig)
    fig_positions(generate_ula(16, LAM / 2), "ULA (M=16)", "fig5_ula_pos.png")
    return rows


def analyze_uca():
    phi = np.linspace(0, 360, 7201)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6),
                             subplot_kw={"projection": "polar"})
    rows = []
    for ax, M in zip(axes, (9, 16)):
        pos = generate_uca(M, LAM)
        B = beampattern(pos, phi, 0.0, LAM)              # θ=0, varre φ
        ax.plot(np.deg2rad(phi), np.clip(B, -40, 0))
        ax.set_ylim(-40, 0); ax.set_title(f"UCA M={M}, R=λ")
        ripple = B.max() - B.min()                       # variação azimutal
        rows.append(("UCA", M, ripple, sll_from_cut(phi, B), directivity(pos)))
    fig.suptitle("Beampattern UCA — corte azimutal θ=0° (pesos uniformes)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig5_uca_bp.png"), dpi=130); plt.close(fig)
    fig_positions(generate_uca(16, LAM), "UCA (M=16, R=λ)", "fig5_uca_pos.png")
    return rows


def analyze_upa():
    phi = np.linspace(-90, 90, 361)
    th = np.linspace(-90, 90, 361)
    PHI, TH = np.meshgrid(phi, th, indexing="xy")
    rows = []
    for M in (3, 4):
        pos = generate_upa(M, M, LAM / 2, LAM / 2)
        B = beampattern(pos, PHI, TH, LAM)
        # mapa de calor
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(PHI, TH, np.clip(B, -40, 0), shading="auto", cmap="viridis")
        ax.set_xlabel("Azimute φ [°]"); ax.set_ylabel("Elevação θ [°]")
        ax.set_title(f"Beampattern UPA {M}×{M} (d=λ/2)")
        fig.colorbar(im, label="Ganho [dB]")
        fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f"fig5_upa{M}_heat.png"), dpi=130); plt.close(fig)
        # superfície 3D
        fig = plt.figure(figsize=(6.5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(PHI, TH, np.clip(B, -40, 0), cmap="viridis",
                        linewidth=0, antialiased=True)
        ax.set_xlabel("φ [°]"); ax.set_ylabel("θ [°]"); ax.set_zlabel("Ganho [dB]")
        ax.set_title(f"Superfície — UPA {M}×{M}")
        fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f"fig5_upa{M}_surf.png"), dpi=130); plt.close(fig)
        # cortes para métricas: φ-cut (θ=0) e θ-cut (φ=0)
        cut_phi = beampattern(pos, phi, 0.0, LAM)
        cut_th = beampattern(pos, 0.0, th, LAM)
        hpbw_az = hpbw_from_cut(phi, cut_phi)
        hpbw_el = hpbw_from_cut(th, cut_th)
        rows.append(("UPA", M * M, (hpbw_az, hpbw_el),
                     sll_from_cut(phi, cut_phi), directivity(pos)))
    fig_positions(generate_upa(4, 4, LAM / 2, LAM / 2), "UPA (4×4)", "fig5_upa_pos.png")
    return rows


def analyze_cyl():
    phi = np.linspace(0, 360, 361)
    th = np.linspace(-90, 90, 181)
    PHI, TH = np.meshgrid(phi, th, indexing="xy")
    rows = []
    for Mc in (9, 16):
        for Nz in (4, 6):
            pos = generate_ucya(Mc, Nz, LAM, LAM / 2)
            B = beampattern(pos, PHI, TH, LAM)
            fig, ax = plt.subplots(figsize=(6.2, 5))
            im = ax.pcolormesh(PHI, TH, np.clip(B, -40, 0), shading="auto", cmap="viridis")
            ax.set_xlabel("Azimute φ [°]"); ax.set_ylabel("Elevação θ [°]")
            ax.set_title(f"Cilíndrico Mc={Mc}, Nz={Nz} (R=λ, dz=λ/2)")
            fig.colorbar(im, label="Ganho [dB]")
            fig.tight_layout()
            fig.savefig(os.path.join(FIGDIR, f"fig5_cyl_{Mc}_{Nz}_map.png"), dpi=130); plt.close(fig)
            cut_th = beampattern(pos, 0.0, th, LAM)
            cut_phi = beampattern(pos, phi, 0.0, LAM)
            rows.append(("Cil.", Mc * Nz,
                         (cut_phi.max() - cut_phi.min(), hpbw_from_cut(th, cut_th)),
                         sll_from_cut(th, cut_th), directivity(pos)))
    fig_positions(generate_ucya(16, 4, LAM, LAM / 2),
                  "Cilíndrico (Mc=16, Nz=4)", "fig5_cyl_pos.png")
    return rows


if __name__ == "__main__":
    print("\n=== ULA (HPBW em θ, SLL, D) ===")
    for r in analyze_ula():
        print(f"  M={r[1]:2d}  HPBW={r[2]:5.2f}°  SLL={r[3]:6.2f} dB  D={r[4]:6.2f}")

    print("\n=== UCA (ripple azimutal, SLL, D) — uniforme => omnidirecional em φ ===")
    for r in analyze_uca():
        print(f"  M={r[1]:2d}  ripple_az={r[2]:5.2f} dB  SLL={r[3]:6.2f} dB  D={r[4]:6.2f}")

    print("\n=== UPA ((HPBW_az,HPBW_el), SLL, D) ===")
    for r in analyze_upa():
        az, el = r[2]
        print(f"  {r[1]:2d} elem  HPBW_az={az:5.2f}°  HPBW_el={el:5.2f}°  "
              f"SLL={r[3]:6.2f} dB  D={r[4]:6.2f}")

    print("\n=== Cilíndrico ((ripple_az,HPBW_el), SLL, D) ===")
    for r in analyze_cyl():
        rip, el = r[2]
        print(f"  {r[1]:2d} elem  ripple_az={rip:5.2f} dB  HPBW_el={el:5.2f}°  "
              f"SLL={r[3]:6.2f} dB  D={r[4]:6.2f}")
