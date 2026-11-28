"""Efecto de la dimensión de estado N en la dinámica de tokens de un SSM selectivo.

En src/ssm.py la dinámica es

    dx_{dl}/dt = sum_{j<=l} P_{dlj}(t) x_{dj}(t),   P_{dlj} ∝ x_l^T M x_j * Delta_d(x_j) * exp(-a_d ...)

con M = S_C^T S_B (D x D).

La dimensión de estado N aparece a través de M, donde S_B, S_C tienen forma (N, D) y por tanto 
rank(M) <= min(N, D). Con D = 1, M es un escalar mu y N solo cambia el valor de mu; el efecto interesante de
N aparece con D > 1 (varios canales), y es el de la pérdida de rango de M.

Escenarios (mismas condiciones que sim_parametrizacion.py, extendidas a D canales con una
perturbación del 5 % entre canales para romper la simetría):

    rapida     S_Delta = +I, x0 positivo creciente (S_Delta x0 > 0)
    lenta      S_Delta = -3 I, x0 = 0.8, 0.9, ... (S_Delta x0 <= -2.12 y decreciente)
    aleatorio  S_Delta = s_delta I (--s-delta), x0 ~ N(0, 1)

Clasificación en t_max, con la pendiente de log ||x_l|| frente a log t en el último 80 % del
intervalo (t >= 0.2 t_max), token a token. Teoría 1D: convergencia -> pendiente -1/2.

    explosión    el simulador detecta explosión en tiempo finito (|x| > limite o no finito)
    convergente  pendiente < -umbral_conv en todos los tokens
    acotado      pendiente en [-umbral_conv, umbral_div] en todos: la norma se estanca
    div. lenta   pendiente > umbral_div en todos (crecimiento lento, tipo (ln t)^l)
    mixto        tokens en clases distintas

Uso:
    python -m experiments.sim_dimension_estado                     # los tres escenarios
    python -m experiments.sim_dimension_estado --escenario lenta --n-seeds 100
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
from src.ssm import norma_por_posicion, simular

FIG_DIR = RAIZ / "figures"
FAMILIAS = ("neg", "pos", "aleatoria")
ESCENARIOS = ("rapida", "lenta", "aleatorio")
TITULOS = {
    "neg": r"$M=-S^\top S$",
    "pos": r"$M=+S^\top S$",
    "aleatoria": r"$M=S_C^\top S_B$",
}
CLASES = (
    ("convergente", "tab:blue"),
    ("acotado", "tab:gray"),
    ("div. lenta", "tab:orange"),
    ("explosión", "tab:red"),
    ("mixto", "tab:purple"),
)

# Tokens iniciales de sim_parametrizacion.py (para la verificación con D = 1)
X0_CONV = np.array([1.0, -0.7, 0.5, 0.9, -1.2, 0.6])
X0_LENTA = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
X0_RAPIDA = np.array([0.5, 0.7, 0.9, 1.0, 1.1, 1.3])


# --------------------------------------------------------------------------- #
# Construcción de M y condiciones iniciales
# --------------------------------------------------------------------------- #
def construir_M(familia, N, D, rng):
    """Matriz M (D, D) de rango <= N, con escala independiente de N."""
    if familia in ("neg", "pos"):
        S = rng.standard_normal((N, D)) / np.sqrt(N)  # E[tr(S^T S)]/D = 1
        return (-1.0 if familia == "neg" else 1.0) * (S.T @ S)
    sB = rng.standard_normal((N, D)) * N ** -0.25
    sC = rng.standard_normal((N, D)) * N ** -0.25
    return sC.T @ sB


def condiciones_iniciales(escenario, L, D, rng, s_delta_aleatorio=1.0):
    """Devuelve (x0 de forma (L, D), s_delta) del escenario."""
    if escenario == "aleatorio":
        return rng.standard_normal((L, D)), s_delta_aleatorio
    if escenario == "rapida":
        base, s = np.linspace(0.5, 1.3, L), 1.0
    elif escenario == "lenta":
        base, s = 0.8 + 0.1 * np.arange(L), -3.0
    else:
        raise ValueError(escenario)
    return base[:, None] * (1.0 + 0.05 * rng.standard_normal((L, D))), s


# --------------------------------------------------------------------------- #
# Simulación y clasificación
# --------------------------------------------------------------------------- #
def correr(M, x0, s_delta, args):
    D = M.shape[0]
    return simular(
        x0, M, s_delta * np.eye(D), args.a * np.ones(D), args.t_max,
        dt=args.dt, limite=args.limite, paso_relativo=args.paso_relativo,
    )


def pendientes(t, X, frac=0.2):
    """Pendiente de log ||x_l|| frente a log t para t >= frac * t_final, token a token."""
    n = np.maximum(norma_por_posicion(X), 1e-300)
    sel = t >= frac * t[-1]
    if sel.sum() < 3:
        return np.full(n.shape[1], np.nan)
    lt = np.log(t[sel])
    return np.array([np.polyfit(lt, np.log(n[sel, l]), 1)[0] for l in range(n.shape[1])])


def clasificar(t, X, exploto, args):
    """Devuelve (clase, pendientes por token o None)."""
    if exploto:
        return "explosión", None
    s = pendientes(t, X)
    c = np.where(s < -args.umbral_conv, 0, np.where(s > args.umbral_div, 2, 1))
    nombres = ("convergente", "acotado", "div. lenta")
    if np.all(c == c[0]):
        return nombres[c[0]], s
    return "mixto", s


# --------------------------------------------------------------------------- #
# Experimentos
# --------------------------------------------------------------------------- #
def experimento_trayectorias(esc, args):
    dims = args.dims_tray
    x0, s = condiciones_iniciales(esc, args.L, args.D, np.random.default_rng(args.seed), args.s_delta)
    colores = plt.cm.viridis(np.linspace(0, 0.9, args.L))

    fig, ax = plt.subplots(len(FAMILIAS), len(dims), figsize=(4.2 * len(dims), 3.2 * len(FAMILIAS)),
                           sharey="row", squeeze=False)
    for i, fam in enumerate(FAMILIAS):
        for j, N in enumerate(dims):
            M = construir_M(fam, N, args.D, np.random.default_rng(args.seed + N))
            t, X, exploto = correr(M, x0, s, args)
            n = norma_por_posicion(X)
            for l in range(args.L):
                ax[i, j].plot(t, n[:, l], color=colores[l], label=f"token {l + 1}")
            ax[i, j].set_xscale("log")
            ax[i, j].set_yscale("log")

            clase, _ = clasificar(t, X, exploto, args)
            ax[i, j].set_title(f"{TITULOS[fam]}, N={N} ({clase})", fontsize=9)
            ax[i, j].grid(alpha=0.3)
        ax[i, 0].set_ylabel(r"$\|x_l(t)\|$")
    for j in range(len(dims)):
        ax[-1, j].set_xlabel("t")
    ax[0, 0].legend(fontsize=6)
    fig.suptitle(f"Escenario {esc}: D={args.D}, L={args.L}, a={args.a}", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / f"dim_{esc}_trayectorias.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[{esc}] guardada {out}")

# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--escenario", choices=ESCENARIOS + ("todos",), default="todos")
    p.add_argument("--dims", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32], help="dimensiones de estado N")
    p.add_argument("--dims-tray", type=int, nargs="+", default=[1, 4, 16], help="N de la figura de trayectorias")
    p.add_argument("--D", type=int, default=8, help="número de canales")
    p.add_argument("--L", type=int, default=6, help="número de tokens")
    p.add_argument("--t-max", type=float, default=1000.0)
    p.add_argument("--dt", type=float, default=5e-3)
    p.add_argument("--paso-relativo", type=float, default=0.02)
    p.add_argument("--limite", type=float, default=1e6, help="umbral de explosión en |x|")
    p.add_argument("--s-delta", type=float, default=1.0, help="S_Delta = s_delta I (solo escenario aleatorio)")
    p.add_argument("--a", type=float, default=1.0, help="A_d = -a I")
    p.add_argument("--umbral-conv", type=float, default=0.2, help="pendiente < -umbral -> convergente")
    p.add_argument("--umbral-div", type=float, default=0.1, help="pendiente > umbral -> divergente lenta")
    p.add_argument("--n-seeds", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sin-verificar", action="store_true", help="omitir la verificación con D = 1")
    args = p.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for esc in (ESCENARIOS if args.escenario == "todos" else (args.escenario,)):
        experimento_trayectorias(esc, args)


if __name__ == "__main__":
    main()