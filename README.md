# Modelagem e Análise de Arranjos de Sensores

Trabalho da disciplina **Processamento de Sinais I** (Prof. Rafael S. Chaves).
Implementação, em Python, da modelagem e análise de arranjos de sensores:
geometrias, vetor diretor, beampattern, beamforming e transmissão direcional.

## Integrantes

- [Nome do integrante 1]
- [Nome do integrante 2]
- [Nome do integrante 3]

## Descrição do projeto

O projeto implementa os blocos fundamentais do processamento espacial de sinais:

- geração de quatro geometrias de arranjos: linear (ULA), circular (UCA),
  planar (UPA) e cilíndrica (UCyA);
- cálculo do vetor diretor (*steering vector*) e do diagrama de radiação
  (*beampattern*) normalizado em dB;
- beamformer convencional *Delay-and-Sum*;
- experimento de transmissão direcional entre dois arranjos lineares,
  avaliando potência recebida, ganho do enlace e correlação;
- análise da separação de múltiplas fontes em função do número de sensores
  e do espaçamento.

Todas as figuras do artigo são geradas automaticamente pelos scripts deste
repositório, sem edição gráfica manual.

## Estrutura dos diretórios

```
Projeto_Arranjos/
├── article/        # artigo final em PDF (formato IEEE)
├── src/            # código-fonte
│   ├── arrays.py                  # funções fundamentais (núcleo)
│   ├── analysis_section5.py       # diagramas de radiação + métricas
│   ├── analysis_section7.py       # transmissão direcional
│   └── analysis_section7_extra.py # múltiplos sinais (extra)
├── figures/        # figuras geradas (saída)
├── data/           # dados intermediários (se houver)
├── examples/       # exemplos de uso das funções
│   ├── example_geometrias.py
│   ├── example_beampattern.py
│   └── example_beamformer.py
├── main.py         # reproduz TODOS os resultados de uma vez
├── requirements.txt
└── README.md
```

## Dependências

- Python 3.9+
- NumPy
- Matplotlib

Instale com:

```bash
pip install -r requirements.txt
```

## Instruções de execução

Para reproduzir **todas** as figuras do artigo (geradas em `figures/`):

```bash
python main.py
```

Para rodar um exemplo isolado:

```bash
python examples/example_geometrias.py
python examples/example_beampattern.py
python examples/example_beamformer.py
```

## Funções principais (`src/arrays.py`)

| Função | Descrição |
|--------|-----------|
| `generate_ula(M, d)` | gera as posições de um arranjo linear |
| `generate_uca(M, R)` | gera as posições de um arranjo circular |
| `generate_upa(Mx, My, dx, dy)` | gera as posições de um arranjo planar |
| `generate_ucya(Mc, Nz, R, dz)` | gera as posições de um arranjo cilíndrico |
| `steering_vector(positions, azimuth, elevation, wavelength)` | vetor diretor |
| `beampattern(positions, azimuth, elevation, wavelength, weights)` | diagrama de radiação (dB) |
| `beamformer(x, positions, steering_direction, wavelength)` | beamformer Delay-and-Sum |

Os ângulos são fornecidos em graus.
