"""
section7.py
===========
Seção 7 — Transmissão Direcional (enlace Tx/Rx em linha de visada).

Cenário:
  - Tx: ULA de 8 elementos, d=λ/2, apontando para θ_T = 20°
  - Rx: ULA de 8 elementos, d=λ/2
  - Propagação LOS; ângulo de chegada no Rx = θ_link = 20° (fixo)
  - Sinal: m(t) = sin(2π·500·t) + 0.5·sin(2π·1500·t)
  - f_c = 1 GHz (frequência de operação) -> λ = c/f_c = 0.3 m, d = 0.15 m

Modelo de banda estreita (B≈1.5 kHz << f_c=1 GHz):
  o atraso entre elementos vira defasagem da portadora.
  Sinal no elemento m do Rx:
      x_m(t) = G_T · m(t) · exp(-j k z_m sinθ_link) + n_m(t)
  com G_T = |AF_T(θ_link)| (ganho do Tx na direção do enlace; máximo,
  pois o Tx aponta para θ_T = θ_link), e n_m ruído gaussiano complexo.

  Rx delay-and-sum apontando para θ_R:
      y(t) = (1/N_R) Σ_m exp(+j k z_m sinθ_R) x_m(t)
           = (G_T/N_R) m(t) AF_R(θ_link,θ_R) + ruído combinado

Métricas:
  - Potência recebida  P(θ_R) = média|y|²
  - Ganho do enlace    em dB, relativo a 1 elemento sem beamforming
  - Ganho espacial      = N_R coerente no alinhamento
  - Correlação          ρ(θ_R) = |<m,y>|/(||m|| ||y||)
"""

import numpy as np
import os
FIGDIR = os.environ.get("FIGDIR") or os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arrays import generate_ula, steering_vector, array_factor, beamformer, beampattern

# ---------- parâmetros físicos ----------
C = 3e8
FC = 1e9
LAM = C / FC                 # 0.3 m
D = LAM / 2                  # 0.15 m
N = 8
THETA_T = 20.0              # apontamento do Tx
THETA_LINK = 20.0          # direção de chegada no Rx (LOS)

# ---------- sinal m(t) ----------
FS = 16000.0                # taxa de amostragem do banda-base [Hz]
DUR = 0.05                  # duração [s]
t = np.arange(0, DUR, 1 / FS)
m = np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 1500 * t)

# ---------- arranjos ----------
tx = generate_ula(N, D)
rx = generate_ula(N, D)

# fase de chegada por elemento (portadora), direção θ_link, φ=0
a_link = steering_vector(rx, 0.0, THETA_LINK, LAM)          # exp(-j k z_m sinθ_link)

# ganho do Tx na direção do enlace (Tx apontando p/ θ_T)
w_T = steering_vector(tx, 0.0, THETA_T, LAM)
G_T = np.abs(np.conj(w_T) @ steering_vector(tx, 0.0, THETA_LINK, LAM))   # = N (alinhado)

SNR_dB = 10.0               # SNR por elemento (sinal vs ruído)
rng = np.random.default_rng(0)


def received_signals(theta_link=THETA_LINK, add_noise=True, sig=m):
    """Gera x_m(t) nos N elementos do Rx (sinal complexo banda-base)."""
    a = steering_vector(rx, 0.0, theta_link, LAM)            # (N,)
    x = G_T * np.outer(a, sig)                               # (N, T)
    if add_noise:
        p_sig = np.mean(np.abs(G_T * sig) ** 2)
        p_noise = p_sig / (10 ** (SNR_dB / 10))
        noise = np.sqrt(p_noise / 2) * (rng.standard_normal(x.shape)
                                        + 1j * rng.standard_normal(x.shape))
        x = x + noise
    return x


def metrics(theta_R, x):
    """Potência recebida, correlação com m(t)."""
    y = beamformer(x, rx, (0.0, theta_R), LAM)
    P = np.mean(np.abs(y) ** 2)
    # correlação (módulo) entre m(t) real e y(t)
    yc = y - y.mean()
    mc = m - m.mean()
    rho = np.abs(np.vdot(mc, yc)) / (np.linalg.norm(mc) * np.linalg.norm(yc))
    return P, rho, y


# =====================================================================
# (a) Alinhamento perfeito θ_R = 20°
# =====================================================================
x_clean = received_signals(add_noise=False)
P_align, _, y_align = metrics(THETA_T, x_clean)

# potência de referência: 1 elemento, sem beamforming (sinal só)
P_ref = np.mean(np.abs(G_T * a_link[0] * m) ** 2) / (N ** 2)   # 1 elem após /N? definimos abaixo
# referência mais clara: um único sensor recebendo G_T*m, sem ganho de arranjo
P_single = np.mean(np.abs(m) ** 2)            # 1 elemento, ganho Tx normalizado fora
link_gain_align_dB = 10 * np.log10(P_align / P_single)

print("=== (a) Alinhamento perfeito θ_R = 20° ===")
print(f"  Potência recebida P = {P_align:.4f}")
print(f"  Ganho do enlace = {link_gain_align_dB:.2f} dB  (esperado ~20log10(N·G_T)... )")
print(f"  Ganho espacial (coerente) = N = {N}  -> {20*np.log10(N):.2f} dB em amplitude")

# forma de onda recebida vs transmitida (alinhado, com ruído)
x_n = received_signals(add_noise=True)
_, _, y_noisy = metrics(THETA_T, x_n)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t * 1e3, m, label="m(t) transmitido", lw=1.6)
ax.plot(t * 1e3, np.real(y_noisy) / G_T, label="y(t) recebido (norm.)", lw=1.0, alpha=.8)
ax.set_xlim(0, 6); ax.set_xlabel("t [ms]"); ax.set_ylabel("amplitude")
ax.set_title("Alinhamento perfeito θ_R=20°: forma de onda recebida")
ax.legend(); ax.grid(True, alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig7_waveform_align.png"), dpi=130); plt.close(fig)


# =====================================================================
# (b) Desalinhamento angular (conjunto discreto)
# =====================================================================
theta_set = [-90, -60, -30, 0, 10, 20, 30, 60, 90]
print("\n=== (b) Desalinhamento angular ===")
print(f"  {'θ_R':>5} | {'P (norm.)':>10} | {'ganho esp.[dB]':>13} | {'correlação':>10}")
rows = []
for th in theta_set:
    x_n = received_signals(add_noise=True)
    P, rho, _ = metrics(th, x_n)
    P_norm = P / P_align
    gain_dB = 10 * np.log10(P_norm)
    rows.append((th, P_norm, gain_dB, rho))
    print(f"  {th:5d} | {P_norm:10.4f} | {gain_dB:13.2f} | {rho:10.4f}")


# =====================================================================
# (c) Varredura contínua -90..90 + comparação com beampattern
# =====================================================================
theta_R = np.linspace(-90, 90, 721)
P_sweep = np.zeros_like(theta_R)
for i, th in enumerate(theta_R):
    x_n = received_signals(add_noise=False)        # sem ruído p/ curva limpa
    P_sweep[i], _, _ = metrics(th, x_n)
P_sweep_dB = 10 * np.log10(P_sweep / P_sweep.max())

# beampattern do Rx (referência), avaliado na chegada θ_link variando θ_R:
# B_R(θ_R) = |AF_R(θ_link, steered θ_R)|  -> aqui equivale a olhar o
# beampattern do ULA Rx em torno de θ_link. Calculamos diretamente:
B_ref = np.zeros_like(theta_R)
for i, th in enumerate(theta_R):
    w = steering_vector(rx, 0.0, th, LAM)
    AF = np.conj(w) @ steering_vector(rx, 0.0, THETA_LINK, LAM)
    B_ref[i] = np.abs(AF) / N
B_ref_dB = 20 * np.log10(np.maximum(B_ref, 1e-12))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(theta_R, P_sweep_dB, label="Potência recebida (simulada)", lw=2)
ax.plot(theta_R, B_ref_dB, "--", label="Beampattern do Rx |AF_R|²", lw=1.4)
ax.axvline(THETA_LINK, color="k", ls=":", alpha=.6, label="θ_link = 20°")
ax.set_xlabel("θ_R [°]"); ax.set_ylabel("Ganho normalizado [dB]")
ax.set_title("Potência recebida × θ_R  vs  beampattern do Rx")
ax.set_ylim(-40, 2); ax.grid(True, alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig7_power_vs_angle.png"), dpi=130); plt.close(fig)

# ganho do enlace vs θ_R (em dB, ref. 1 elemento)
link_gain_dB = 10 * np.log10(P_sweep / P_single)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(theta_R, link_gain_dB, color="tab:green", lw=2)
ax.axvline(THETA_LINK, color="k", ls=":", alpha=.6)
ax.set_xlabel("θ_R [°]"); ax.set_ylabel("Ganho do enlace [dB]")
ax.set_title("Ganho do enlace × θ_R (ref.: 1 elemento)")
ax.grid(True, alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig7_linkgain_vs_angle.png"), dpi=130); plt.close(fig)

print("\n  Erro máx. potência-vs-beampattern: "
      f"{np.max(np.abs(P_sweep_dB - B_ref_dB)):.3e} dB (devem coincidir)")


# =====================================================================
# (Extra) Dois sinais — separação angular
# =====================================================================
def two_source_scan(sep, th1=0.0):
    th2 = th1 + sep
    s1 = np.sin(2 * np.pi * 500 * t)
    s2 = np.sin(2 * np.pi * 1500 * t)
    a1 = steering_vector(rx, 0.0, th1, LAM)
    a2 = steering_vector(rx, 0.0, th2, LAM)
    x = G_T * (np.outer(a1, s1) + np.outer(a2, s2))
    scan = np.linspace(-40, 60, 1001)
    P = np.array([np.mean(np.abs(beamformer(x, rx, (0.0, a), LAM)) ** 2) for a in scan])
    return scan, P / P.max(), th1, th2

fig, ax = plt.subplots(figsize=(9, 4.2))
for sep in (30, 12, 6):
    scan, P, t1, t2 = two_source_scan(sep)
    ax.plot(scan, 10 * np.log10(P), label=f"Δθ = {sep}° (fontes em {t1:.0f}° e {t2:.0f}°)")
ax.set_xlabel("direção de varredura θ [°]"); ax.set_ylabel("potência [dB]")
ax.set_title("Separação angular de 2 fontes (Rx ULA N=8)")
ax.set_ylim(-30, 2); ax.grid(True, alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig7_two_sources.png"), dpi=130); plt.close(fig)

print("\nFiguras da Seção 7 geradas.")
