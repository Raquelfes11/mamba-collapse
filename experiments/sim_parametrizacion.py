"""Experimento de parametrización del SSM selectivo en el caso unidimensional.

Barre el signo de mu, el decaimiento a y la escala S_Delta, y dibuja la norma
de cada token en función del tiempo (escala logarítmica). Las figuras se
guardan en la carpeta figures del repositorio.

Uso, desde la raíz del repositorio:
    python experiments/sim_parametrizacion.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

from ssm import clasificar_1d, norma_por_posicion, simular_1d  # noqa: E402

CARPETA_FIGURAS = os.path.join(RAIZ, "figures")
L = 6  # número de tokens

# Tokens iniciales de cada escenario. Para la divergencia lenta se pide
# S_Delta * x0 <= -2.12 y decreciente con la posición.
X0_CONV = np.array([1.0, -0.7, 0.5, 0.9, -1.2, 0.6])
X0_LENTA = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
X0_RAPIDA = np.array([0.5, 0.7, 0.9, 1.0, 1.1, 1.3])

# (nombre, mu, s_delta, x0, t_max, dt)
ESCENARIOS = [
    ("convergencia: mu = -1", -1.0, 1.0, X0_CONV, 50.0, 5e-3),
    ("divergencia lenta: mu = 1, S_Delta x0 < 0", 1.0, -3.0, X0_LENTA, 50.0, 5e-3),
    ("divergencia rápida: mu = 1, S_Delta x0 > 0", 1.0, 1.0, X0_RAPIDA, 5.0, 1e-3),
]


def correr(mu, s_delta, a, x0, t_max, dt):
    """Simula y devuelve el tiempo, la norma por posición y si hubo explosión."""
    t, X, exploto = simular_1d(
        x0, mu, s_delta, a, t_max, dt=dt, limite=1e6, cada=max(1, int(t_max / dt / 2000))
    )
    return t, norma_por_posicion(X), exploto


def dibujar(ax, t, normas, titulo):
    """Dibuja la norma de cada token, coloreando por posición."""
    colores = plt.cm.viridis(np.linspace(0, 0.9, normas.shape[1]))
    for l in range(normas.shape[1]):
        ax.plot(t, normas[:, l], color=colores[l], label=f"token {l + 1}")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_title(titulo, fontsize=9)
    ax.grid(alpha=0.3)


def figura_escenarios():
    """Los tres escenarios del caso unidimensional."""
    fig, ejes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (nombre, mu, s, x0, t_max, dt) in zip(ejes, ESCENARIOS):
        t, normas, exploto = correr(mu, s, 1.0, x0, t_max, dt)
        dibujar(ax, t, normas, nombre)
        escenario = clasificar_1d(mu, s, x0)
        if exploto:
            print(f"[escenarios] {nombre}: clasificado como {escenario}, explota hacia t = {t[-1]:.3f}")
        else:
            print(f"[escenarios] {nombre}: clasificado como {escenario}, t final = {t[-1]:.2f}, "
                  f"|x| final por token = {np.round(normas[-1], 3)}")
    ejes[0].set_ylabel("|x_l(t)|")
    ejes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "param_escenarios.png"), dpi=150)
    plt.close(fig)


def figura_barrido_a(valores_a=(0.1, 1.0, 5.0)):
    """Efecto del decaimiento a, con mu fijo, en convergencia y divergencia lenta."""
    fig, ejes = plt.subplots(2, len(valores_a), figsize=(13, 6.5), sharey="row")
    for fila, (nombre, mu, s, x0, t_max, dt) in enumerate(ESCENARIOS[:2]):
        for col, a in enumerate(valores_a):
            t, normas, exploto = correr(mu, s, a, x0, t_max, dt)
            dibujar(ejes[fila, col], t, normas, f"{nombre.split(':')[0]}, a = {a}")
            print(f"[barrido a] {nombre.split(':')[0]}, a = {a}: explota = {exploto}, "
                  f"|x| final por token = {np.round(normas[-1], 3)}")
        ejes[fila, 0].set_ylabel("|x_l(t)|")
    ejes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "param_a.png"), dpi=150)
    plt.close(fig)


def figura_barrido_sdelta(valores_s=(0.5, 1.0, 2.0, 4.0)):
    """Efecto de la escala S_Delta en el caso convergente (mu = -1)."""
    fig, ejes = plt.subplots(1, len(valores_s), figsize=(14, 3.8), sharey=True)
    _, mu, _, x0, t_max, dt = ESCENARIOS[0]
    for ax, s in zip(ejes, valores_s):
        t, normas, exploto = correr(mu, s, 1.0, x0, t_max, dt)
        dibujar(ax, t, normas, f"convergencia, S_Delta = {s}")
        print(f"[barrido S_Delta] S_Delta = {s}: |x| final por token = {np.round(normas[-1], 3)}")
    ejes[0].set_ylabel("|x_l(t)|")
    ejes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "param_sdelta.png"), dpi=150)
    plt.close(fig)


def figura_tasas_largas(t_max=1e8):
    """Comprueba las tasas del teorema a tiempos muy largos, con paso proporcional a t.

    Izquierda: en el caso convergente, sqrt(t) por la norma de cada token debe
    estabilizarse en una constante (tasa 1/sqrt(t)). Derecha: en la divergencia
    lenta, la norma frente a ln t; el teorema da la cota (ln t)^l.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    colores = plt.cm.viridis(np.linspace(0, 0.9, L))

    _, mu, s, x0, _, dt = ESCENARIOS[0]
    t, X, _ = simular_1d(x0, mu, s, 1.0, t_max, dt=dt, paso_relativo=0.01)
    normas = norma_por_posicion(X)
    for l in range(L):
        ax1.plot(t[1:], np.sqrt(t[1:]) * normas[1:, l], color=colores[l], label=f"token {l + 1}")
    ax1.axhline(1 / np.sqrt(2 * np.log(2) * abs(mu)), color="k", ls="--", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel("t")
    ax1.set_ylabel("sqrt(t) |x_l(t)|")
    ax1.set_title("convergencia: sqrt(t) |x| se estabiliza", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    sel = t > 1e4
    pend = [np.polyfit(np.log(t[sel]), np.log(normas[sel, l]), 1)[0] for l in range(L)]
    print(f"[tasas largas] convergencia, pendiente log-log para t > 1e4 = {np.round(pend, 3)} (teorema: -0.5)")

    _, mu, s, x0, _, dt = ESCENARIOS[1]
    t, X, _ = simular_1d(x0, mu, s, 1.0, t_max, dt=dt, paso_relativo=0.01, limite=1e12)
    normas = norma_por_posicion(X)
    for l in range(L):
        ax2.plot(np.log(t[1:]), normas[1:, l], color=colores[l], label=f"token {l + 1}")
    ax2.set_xlabel("ln t")
    ax2.set_ylabel("|x_l(t)|")
    ax2.set_title("divergencia lenta: norma frente a ln t", fontsize=9)
    ax2.grid(alpha=0.3)
    sel = t > 1e3
    pend = [np.polyfit(np.log(np.log(t[sel])), np.log(normas[sel, l]), 1)[0] for l in range(L)]
    print(f"[tasas largas] divergencia lenta, exponente de ln t ajustado para t > 1e3 = {np.round(pend, 2)} "
          f"(cota del teorema: l = 1..{L})")

    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_FIGURAS, "param_tasas_largas.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)
    figura_escenarios()
    figura_barrido_a()
    figura_barrido_sdelta()
    figura_tasas_largas()
    print("Figuras guardadas en", CARPETA_FIGURAS)
