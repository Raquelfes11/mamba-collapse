"""Experimento de longitud de secuencia y memoria de largo contexto (caso unidimensional).

Se mide cuánta memoria del principio de la secuencia conserva el último token
al crecer el número de tokens L, en los tres escenarios del paper. Hay dos
medidas:

A. Peso del primer token sobre el último, |P_{L1}(t)|, la entrada de la matriz
   de atención oculta que conecta la posición L con la posición 1.
B. Sensibilidad del último token al primero, |dx_L(T) / dx_1(0)|. Se calcula
   con el método del paso complejo (se perturba x_1(0) con una parte imaginaria
   de 1e-30), que da la derivada exacta sin el ruido de una diferencia finita.

Además se mira cómo cambia el tiempo de explosión de la divergencia rápida al
aumentar L. Las figuras se guardan en la carpeta figures del repositorio.

Uso, desde la raíz del repositorio:
    python experiments/sim_longitud_secuencia.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

from ssm import matriz_P, simular_1d, softplus  # noqa: E402

CARPETA_FIGURAS = os.path.join(RAIZ, "figures")
LONGITUDES = [2, 4, 8, 16, 32, 64]
VALORES_A = [0.1, 1.0, 5.0]
H_COMPLEJO = 1e-30
T_EVAL = 10.0  # instante de evaluación en convergencia y divergencia lenta
FRACCION_EXPLOSION = 0.8  # en la divergencia rápida se evalúa a esta fracción del tiempo de explosión

ESCENARIOS = ["convergencia", "divergencia lenta", "divergencia rápida"]


def parametros(escenario, L):
    """Devuelve (mu, s_delta, x0, t_max, dt) del escenario con L tokens.

    En la divergencia lenta se cumple S_Delta x_{l0} <= ... <= S_Delta x_{10} <= -2.12,
    con x0 creciente y S_Delta = -3. En la rápida todos los tokens son positivos
    y S_Delta > 0, así que S_Delta x_{l0} > 0 para todo l.
    """
    if escenario == "convergencia":
        return -1.0, 1.0, np.ones(L), T_EVAL, 1e-2
    if escenario == "divergencia lenta":
        x0 = 0.8 + 0.02 * np.arange(L)
        return 1.0, -3.0, x0, T_EVAL, 1e-2
    x0 = np.linspace(0.5, 1.3, L)
    return 1.0, 1.0, x0, 1.0, 1e-3


def correr(mu, s_delta, a, x0, t_max, dt):
    """Simula guardando todos los pasos y devuelve el tiempo, los tokens y si hubo explosión."""
    t, X, exploto = simular_1d(x0, mu, s_delta, a, t_max, dt=dt, limite=1e6, cada=1)
    return t, X[:, :, 0], exploto


def peso_primero_ultimo(X_i, mu, s_delta, a):
    """Devuelve |P_{L1}| para un estado X_i de forma (L,)."""
    P = matriz_P(X_i.reshape(-1, 1), np.array([[mu]]), np.array([[s_delta]]), np.array([a]))
    return abs(P[0, -1, 0])


def medir(escenario, L, a):
    """Calcula las medidas A y B y el tiempo de explosión para un escenario, L y a."""
    mu, s, x0, t_max, dt = parametros(escenario, L)
    t, X, exploto = correr(mu, s, a, x0, t_max, dt)

    t_explosion = t[-1] if exploto else np.nan
    T = FRACCION_EXPLOSION * t[-1] if exploto else T_EVAL
    i_T = int(np.searchsorted(t, T, side="right") - 1)

    x0_c = x0.astype(complex)
    x0_c[0] += 1j * H_COMPLEJO
    t2, X2, _ = simular_1d(x0_c, mu, s, a, t_max, dt=dt, limite=1e6, cada=1)
    i_T = min(i_T, len(t2) - 1)

    return {
        "P_inicio": peso_primero_ultimo(X[0], mu, s, a),
        "P_T": peso_primero_ultimo(X[i_T], mu, s, a),
        "sensibilidad": abs(X2[i_T, -1, 0].imag) / H_COMPLEJO,
        "suma_delta": softplus(s * X[i_T, 1:]).sum(),
        "t_explosion": t_explosion,
        "T": t[i_T],
    }


def figura(resultados, clave, claves_extra, titulo, nombre, ylabel):
    """Dibuja una medida frente a L, una columna por escenario y una curva por a."""
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, esc in zip(ejes, ESCENARIOS):
        for a, color in zip(VALORES_A, plt.cm.viridis([0.1, 0.5, 0.85])):
            ys = [resultados[(esc, L, a)][clave] for L in LONGITUDES]
            ax.plot(LONGITUDES, ys, "o-", color=color, label=f"a = {a}")
            for extra, estilo in claves_extra:
                ys2 = [resultados[(esc, L, a)][extra] for L in LONGITUDES]
                ax.plot(LONGITUDES, ys2, estilo, color=color)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("L (número de tokens)")
        ax.set_title(esc, fontsize=10)
        ax.grid(alpha=0.3)
    ejes[0].set_ylabel(ylabel)
    ejes[0].legend(fontsize=8)
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, nombre), dpi=150)
    plt.close(fig)


def figura_explosion(resultados):
    """Tiempo de explosión de la divergencia rápida frente a L."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for a, color in zip(VALORES_A, plt.cm.viridis([0.1, 0.5, 0.85])):
        ys = [resultados[("divergencia rápida", L, a)]["t_explosion"] for L in LONGITUDES]
        ax.plot(LONGITUDES, ys, "o-", color=color, label=f"a = {a}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("L (número de tokens)")
    ax.set_ylabel("tiempo de explosión")
    ax.set_title("divergencia rápida: tiempo de explosión", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "long_explosion.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)
    resultados = {}
    for esc in ESCENARIOS:
        for a in VALORES_A:
            for L in LONGITUDES:
                resultados[(esc, L, a)] = medir(esc, L, a)
            fila = [resultados[(esc, L, a)] for L in LONGITUDES]
            print(f"[{esc}, a = {a}]")
            print("  |P_L1| en t = 0:     ", np.array([f["P_inicio"] for f in fila]))
            print("  |P_L1| en T:         ", np.array([f["P_T"] for f in fila]))
            print("  sensibilidad en T:   ", np.array([f["sensibilidad"] for f in fila]))
            print("  suma de Delta (k>=2):", np.array([f["suma_delta"] for f in fila]))
            if esc == "divergencia rápida":
                print("  t de explosión:      ", np.array([f["t_explosion"] for f in fila]))

    figura(resultados, "P_T", [("P_inicio", "--")],
           "Peso del primer token sobre el último, |P_L1| (trazo continuo en T, discontinuo en t = 0)",
           "long_peso.png", "|P_L1|")
    figura(resultados, "sensibilidad", [],
           "Sensibilidad del último token al primero en t = 0, |dx_L(T) / dx_1(0)|",
           "long_sensibilidad.png", "|dx_L(T) / dx_1(0)|")
    figura_explosion(resultados)
    print("Figuras guardadas en", CARPETA_FIGURAS)
