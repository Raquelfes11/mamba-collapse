"""Pruebas del simulador base del SSM selectivo.

Comprueban que la versión vectorizada coincide con un cálculo directo de las
fórmulas, que el integrador converge al reducir el paso y que se reproducen
las tasas del caso unidimensional.

Uso, desde la raíz del repositorio:
    python -m pytest tests
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ssm import campo_vectorial, clasificar_1d, simular_1d, softplus  # noqa: E402

X0_CONV = np.array([1.0, -0.7, 0.5, 0.9, -1.2, 0.6])
X0_LENTA = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
X0_RAPIDA = np.array([0.5, 0.7, 0.9, 1.0, 1.1, 1.3])


def campo_directo(x, M, s_delta, a):
    """Calcula dx/dt con bucles, escribiendo las fórmulas tal cual."""
    L, D = x.shape
    dx = np.zeros_like(x)
    for d in range(D):
        delta = [np.log1p(np.exp(s_delta[d] @ x[k])) for k in range(L)]
        for l in range(L):
            for j in range(l + 1):
                g = x[l] @ M @ x[j]
                if j == l:
                    P = g * delta[l]
                else:
                    P = g * delta[j] * np.exp(-a[d] * sum(delta[j + 1:l + 1]))
                dx[l, d] += P * x[j, d]
    return dx


def test_vectorizado_coincide_con_bucle_directo():
    rng = np.random.default_rng(0)
    for D in (1, 3):
        x = rng.normal(size=(5, D))
        M = rng.normal(size=(D, D))
        s = rng.normal(size=(D, D))
        a = rng.uniform(0.1, 3, D)
        assert np.allclose(campo_vectorial(x, M, s, a), campo_directo(x, M, s, a), atol=1e-10)


def test_un_token_coincide_con_referencia_de_paso_fino():
    # Con L = 1 la ecuacion es dx/dt = mu x^3 softplus(s x)
    mu, s, x, n, t_max = -1.0, 1.0, 1.0, 100000, 10.0
    h = t_max / n
    f = lambda y: mu * y**3 * softplus(s * y)
    for _ in range(n):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    _, X, _ = simular_1d([1.0], mu, s, 1.0, t_max, dt=5e-3, cada=10**9)
    assert abs(X[-1, 0, 0] - x) < 1e-8


def test_convergencia_al_reducir_el_paso():
    for x0, mu, s in ((X0_CONV, -1.0, 1.0), (X0_LENTA, 1.0, -3.0)):
        _, grueso, _ = simular_1d(x0, mu, s, 1.0, 20.0, dt=2e-2, cada=10**9)
        _, fino, _ = simular_1d(x0, mu, s, 1.0, 20.0, dt=2.5e-3, cada=10**9)
        assert np.max(np.abs(grueso[-1] - fino[-1])) < 1e-8


def test_escenarios_clasificados_y_comportamiento():
    # Convergencia: tasa 1/sqrt(t) a tiempos largos
    assert clasificar_1d(-1.0, 1.0, X0_CONV) == "convergencia"
    t, X, explota = simular_1d(X0_CONV, -1.0, 1.0, 1.0, 1e6, dt=5e-3, paso_relativo=0.01)
    assert not explota
    sel = t > 1e3
    pendiente = np.polyfit(np.log(t[sel]), np.log(np.abs(X[sel, 0, 0])), 1)[0]
    assert abs(pendiente + 0.5) < 0.01

    # Divergencia lenta: crece sin explotar
    assert clasificar_1d(1.0, -3.0, X0_LENTA) == "divergencia_lenta"
    t, X, explota = simular_1d(X0_LENTA, 1.0, -3.0, 1.0, 1e6, dt=5e-3, paso_relativo=0.01, limite=1e12)
    assert not explota
    assert np.all(np.abs(X[-1, :, 0]) > np.abs(X[0, :, 0]))

    # Divergencia rapida: explota en tiempo finito y el ultimo token es el primero
    assert clasificar_1d(1.0, 1.0, X0_RAPIDA) == "divergencia_rapida"
    t, X, explota = simular_1d(X0_RAPIDA, 1.0, 1.0, 1.0, 5.0, dt=5e-4, limite=1e6, cada=10**9)
    assert explota
    assert t[-1] < 0.2
    assert np.argmax(np.abs(X[-1, :, 0])) == len(X0_RAPIDA) - 1


def test_paso_complejo_coincide_con_diferencia_finita():
    # Derivada de x_L(T) respecto de x_1(0), con tres tokens
    x0 = np.array([0.8, 0.9, 1.0])
    T, dt = 2.0, 1e-2
    h = 1e-30
    x0c = x0.astype(complex)
    x0c[0] += 1j * h
    _, Xc, _ = simular_1d(x0c, 1.0, -3.0, 1.0, T, dt=dt, cada=10**9)
    derivada_compleja = Xc[-1, -1, 0].imag / h

    eps = 1e-6
    _, mas, _ = simular_1d(x0 + np.array([eps, 0, 0]), 1.0, -3.0, 1.0, T, dt=dt, cada=10**9)
    _, menos, _ = simular_1d(x0 - np.array([eps, 0, 0]), 1.0, -3.0, 1.0, T, dt=dt, cada=10**9)
    derivada_finita = (mas[-1, -1, 0] - menos[-1, -1, 0]) / (2 * eps)

    assert abs(derivada_compleja - derivada_finita) < 1e-6 * max(1.0, abs(derivada_finita))
