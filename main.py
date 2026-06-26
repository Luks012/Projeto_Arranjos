#!/usr/bin/env python3
"""
main.py — Reproduz automaticamente todos os resultados do trabalho.

Executa as três análises (Seção 5, Seção 7 e o extra de múltiplos sinais),
gerando todas as figuras do artigo no diretório figures/.

Uso:
    python main.py
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

SCRIPTS = [
    ("Seção 5 — Diagramas de radiação", "analysis_section5.py"),
    ("Seção 7 — Transmissão direcional", "analysis_section7.py"),
    ("Seção 7 — Múltiplos sinais (extra)", "analysis_section7_extra.py"),
]


def main():
    env = {**os.environ, "FIGDIR": FIGDIR}
    for titulo, script in SCRIPTS:
        print("\n" + "=" * 60)
        print(titulo)
        print("=" * 60)
        subprocess.run([sys.executable, os.path.join(SRC, script)],
                       env=env, check=True)
    figs = sorted(f for f in os.listdir(FIGDIR) if f.endswith(".png"))
    print("\n" + "=" * 60)
    print(f"Concluído. {len(figs)} figuras geradas em figures/:")
    for f in figs:
        print("  -", f)


if __name__ == "__main__":
    main()
