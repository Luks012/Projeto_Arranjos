"""
arrays.py
=========
Núcleo de modelagem de arranjos de sensores para Processamento de Sinais I.

Convenção de ângulos (segue EXATAMENTE a Eq. (2) do enunciado):

    u(θ, φ) = [ cosθ cosφ ,  cosθ sinφ ,  sinθ ]^T

onde:
    φ = azimute  (varre o plano xy)
    θ = elevação (componente em z é sinθ)

    u aponta para +x quando (θ=0, φ=0)
    u aponta para +y quando (θ=0, φ=90°)
    u aponta para +z quando (θ=90°)

Número de onda: k = 2π/λ
Vetor diretor:  a_m = exp(-j k r_m·u)            (Eq. 4)
Fator arranjo:  AF = w^H a                        (Eq. 6)
Beampattern:    B = |AF| / max|AF| ;  B_dB = 20log10(B)   (Eqs. 7-8)

Todos os ângulos das funções são em GRAUS (convertidos internamente).
"""

import numpy as np


# =====================================================================
# 4. GERAÇÃO DAS GEOMETRIAS
# =====================================================================

def generate_ula(M, d):
    """
    Arranjo Linear Uniforme (ULA).

    M sensores igualmente espaçados ao longo do eixo z, centrados na origem.

    Por que o eixo z? Com a convenção da Eq. (2), a componente em z de u é
    sinθ. Logo, para um ULA em z, a fase de cada sensor vale k*z_m*sinθ —
    exatamente a expressão clássica de ULA (ψ = k d sinθ). Isso coloca o
    lóbulo principal (pesos uniformes) em broadside θ = 0°, o centro da
    faixa de varredura pedida (-90° ≤ θ ≤ 90°).

    Parâmetros
    ----------
    M : int     número de sensores
    d : float   espaçamento entre sensores [m]

    Retorna
    -------
    positions : ndarray (M, 3)
    """
    m = np.arange(M)
    z = (m - (M - 1) / 2.0) * d          # centraliza na origem
    positions = np.zeros((M, 3))
    positions[:, 2] = z
    return positions


def generate_uca(M, R):
    """
    Arranjo Circular Uniforme (UCA).

    M sensores distribuídos uniformemente em um círculo de raio R no
    plano xy (z = 0). O sensor m fica no ângulo 2π m / M.

    Como o círculo está no plano xy, o arranjo é naturalmente varrido
    pelo azimute φ (0° ≤ φ < 360°), conforme pedido.

    Parâmetros
    ----------
    M : int     número de sensores
    R : float   raio do círculo [m]

    Retorna
    -------
    positions : ndarray (M, 3)
    """
    ang = 2 * np.pi * np.arange(M) / M
    positions = np.zeros((M, 3))
    positions[:, 0] = R * np.cos(ang)
    positions[:, 1] = R * np.sin(ang)
    return positions


def generate_upa(Mx, My, dx, dy):
    """
    Arranjo Planar Uniforme (UPA).

    Grade Mx × My posicionada no plano yz (x = 0), centrada na origem:
        - eixo "horizontal" -> y  (Mx elementos, espaçamento dx)
        - eixo "vertical"   -> z  (My elementos, espaçamento dy)

    Por que o plano yz? Assim o broadside (lóbulo principal com pesos
    uniformes) fica em +x, isto é (θ=0, φ=0) — o centro do mapa
    (azimute × elevação), tornando o heatmap e a superfície 3D legíveis.

    Parâmetros
    ----------
    Mx, My : int     nº de sensores horizontal e vertical
    dx, dy : float   espaçamentos horizontal e vertical [m]

    Retorna
    -------
    positions : ndarray (Mx*My, 3)
    """
    iy = np.arange(Mx)
    iz = np.arange(My)
    y = (iy - (Mx - 1) / 2.0) * dx
    z = (iz - (My - 1) / 2.0) * dy
    YY, ZZ = np.meshgrid(y, z, indexing="ij")     # (Mx, My)
    positions = np.zeros((Mx * My, 3))
    positions[:, 1] = YY.ravel()
    positions[:, 2] = ZZ.ravel()
    return positions


def generate_ucya(Mc, Nz, R, dz):
    """
    Arranjo Cilíndrico Uniforme (UCyA).

    Nz anéis (cada um um UCA de Mc sensores, raio R no plano xy)
    empilhados ao longo de z com espaçamento dz, centrados na origem.

    Parâmetros
    ----------
    Mc : int     sensores por anel
    Nz : int     número de anéis
    R  : float   raio do cilindro [m]
    dz : float   espaçamento vertical entre anéis [m]

    Retorna
    -------
    positions : ndarray (Mc*Nz, 3)
    """
    ang = 2 * np.pi * np.arange(Mc) / Mc
    z_levels = (np.arange(Nz) - (Nz - 1) / 2.0) * dz
    positions = np.zeros((Mc * Nz, 3))
    idx = 0
    for zc in z_levels:
        for a in ang:
            positions[idx] = [R * np.cos(a), R * np.sin(a), zc]
            idx += 1
    return positions


# =====================================================================
# 5. VETOR DIRETOR E BEAMPATTERN
# =====================================================================

def _unit_vector(azimuth_deg, elevation_deg):
    """
    Vetor unitário de propagação u (Eq. 2).
    Aceita escalares ou arrays (broadcasting). Retorna array com
    última dimensão = 3.
    """
    phi = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    th = np.deg2rad(np.asarray(elevation_deg, dtype=float))
    phi, th = np.broadcast_arrays(phi, th)         # alinha formatos
    ux = np.cos(th) * np.cos(phi)
    uy = np.cos(th) * np.sin(phi)
    uz = np.sin(th)
    return np.stack([ux, uy, uz], axis=-1)        # (..., 3)


def steering_vector(positions, azimuth, elevation, wavelength):
    """
    Vetor diretor a(θ, φ)  —  Eq. (4).

        a_m = exp(-j k r_m · u),   k = 2π/λ

    Parâmetros
    ----------
    positions : ndarray (M, 3)
    azimuth   : float   azimute φ [graus]   (escalar)
    elevation : float   elevação θ [graus]  (escalar)
    wavelength: float   comprimento de onda λ [m]

    Retorna
    -------
    a : ndarray complexo (M,)
    """
    k = 2 * np.pi / wavelength
    u = _unit_vector(azimuth, elevation)           # (3,)
    phase = positions @ u                          # (M,)  r_m·u
    return np.exp(-1j * k * phase)


def array_factor(positions, azimuth, elevation, wavelength, weights=None):
    """
    Fator de arranjo AF(θ, φ) = w^H a(θ, φ)  —  Eq. (6).

    azimuth/elevation podem ser escalares ou arrays (de qualquer formato
    compatível por broadcasting). Retorna AF complexo no mesmo formato
    da grade de ângulos.
    """
    M = positions.shape[0]
    if weights is None:
        weights = np.ones(M, dtype=complex)        # pesos uniformes (Eq. 9)
    weights = np.asarray(weights, dtype=complex)

    k = 2 * np.pi / wavelength
    u = _unit_vector(azimuth, elevation)           # (..., 3)
    # r_m · u  para todo m e todo ângulo -> (..., M)
    phase = u @ positions.T                        # (..., M)
    a = np.exp(-1j * k * phase)                    # (..., M)
    # AF = w^H a  =  sum_m conj(w_m) * a_m
    AF = a @ np.conj(weights)                       # (...,)
    return AF


def beampattern(positions, azimuth, elevation, wavelength, weights=None):
    """
    Beampattern normalizado em dB  —  Eqs. (7)-(8).

        B    = |AF| / max|AF|
        B_dB = 20 log10(B)

    Parâmetros
    ----------
    positions : ndarray (M, 3)
    azimuth   : float ou ndarray   azimute(s) φ [graus]
    elevation : float ou ndarray   elevação(ões) θ [graus]
    wavelength: float
    weights   : ndarray (M,) ou None (uniforme)

    Retorna
    -------
    B_dB : ndarray   ganho normalizado em dB (mesmo formato dos ângulos)
    """
    AF = array_factor(positions, azimuth, elevation, wavelength, weights)
    mag = np.abs(AF)
    mag_max = np.max(mag)
    B = mag / mag_max
    # piso para evitar log(0)
    B = np.maximum(B, 1e-12)
    return 20 * np.log10(B)


# =====================================================================
# 6. BEAMFORMING — Delay-and-Sum convencional
# =====================================================================

def beamformer(x, positions, steering_direction, wavelength):
    """
    Beamformer convencional (Delay-and-Sum).

    No modelo de banda estreita, o atraso de propagação entre elementos
    vira uma defasagem pura ditada pela portadora. O beamformer aplica a
    defasagem conjugada (alinha as fases na direção de observação) e soma:

        y(t) = (1/M) * w^H x(t),   com  w = a(steering_direction)

    O fator 1/M dá ganho unitário na direção de apontamento (uma onda
    vinda exatamente dessa direção sai com a mesma amplitude do elemento).

    Parâmetros
    ----------
    x : ndarray (M, T)         sinais recebidos pelos M sensores (T amostras)
    positions : ndarray (M, 3)
    steering_direction : (azimute, elevação) em graus
    wavelength : float

    Retorna
    -------
    y : ndarray (T,)           sinal após o beamformer
    """
    az, el = steering_direction
    w = steering_vector(positions, az, el, wavelength)     # (M,)  w = a(θ,φ)
    M = positions.shape[0]
    y = (np.conj(w) @ x) / M                                # (T,)
    return y
