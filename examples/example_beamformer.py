"""
example_beamformer.py — Demonstra o beamformer Delay-and-Sum.

Gera o sinal recebido por um ULA a partir de uma fonte e mostra que a saída
do beamformer é máxima quando aponta para a direção correta.
Execute a partir da raiz do projeto:  python examples/example_beamformer.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arrays import generate_ula, steering_vector, beamformer

LAM = 0.3
ula = generate_ula(8, LAM / 2)

# sinal e fonte chegando de 25°
t = np.arange(0, 0.02, 1 / 16000)
s = np.sin(2 * np.pi * 500 * t)
a = steering_vector(ula, 0.0, 25.0, LAM)
x = np.outer(a, s)                      # sinais nos 8 sensores

print("Potência de saída do beamformer para diferentes apontamentos:")
for th in (0, 10, 20, 25, 30, 45):
    y = beamformer(x, ula, (0.0, th), LAM)
    P = np.mean(np.abs(y) ** 2)
    marca = "  <-- fonte real" if th == 25 else ""
    print(f"  θ = {th:3d}° : potência = {P:.4f}{marca}")
