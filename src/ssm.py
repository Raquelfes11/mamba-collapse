"""Simulador base de la dinámica de tokens de un SSM selectivo (tipo Mamba).

Implementa el límite continuo en profundidad que estudian Vo, Pham, Tong y
Nguyen (ICLR 2025): se quedan solo con la capa S6 y la conexión residual, y
tratan el índice de capa como tiempo continuo t. Para cada canal d y cada
posición l:

    dx_{dl}/dt = suma_{j <= l} P_{dlj}(t) x_{dj}(t)

    P_{dll} = x_l^T M x_l * Delta_d(x_l)
    P_{dlj} = x_l^T M x_j * Delta_d(x_j) * exp(-a_d * suma_{k=j+1}^{l} Delta_d(x_k))   (j < l)

con M = S_C^T S_B (matriz entrada-salida), Delta_d(u) = softplus(S_{Delta,d} u)
y A_d = -a_d I con a_d > 0.

Nota: las fórmulas de arriba están tomadas de una lectura del paper y hay que
comprobarlas contra el PDF antes de usarlas en el documento.

En el caso unidimensional (D = 1) M es un escalar mu y todo depende de su
signo y del signo de S_Delta * x_{l0}.
"""

import numpy as np


def softplus(u):
    """Calcula softplus(u) = ln(1 + e^u) de forma estable numéricamente."""
    return np.logaddexp(0.0, u)


def matriz_P(x, M, s_delta, a):
    """Devuelve las matrices P de cada canal, de forma (D, L, L).

    x tiene forma (L, D): L tokens con D canales. M es la matriz de entrada
    y salida (D, D), s_delta tiene forma (D, D) (fila d = S_{Delta,d}) y a es
    un vector (D,) con los a_d positivos. P[d] es triangular inferior.
    """
    L, D = x.shape
    G = x @ M @ x.T

    delta = softplus(x @ s_delta.T)

    c = np.cumsum(delta, axis=0)
    dif = c[:, None, :] - c[None, :, :]          # (L, L, D), indices l, j, d
    triangulo = np.tri(L, dtype=bool)[:, :, None]
    dif = np.where(triangulo, dif, 0.0)
    decaimiento = np.exp(-a[None, None, :] * dif)

    P = G[:, :, None] * delta[None, :, :] * decaimiento
    P = np.where(triangulo, P, 0.0)
    return np.transpose(P, (2, 0, 1))


def campo_vectorial(x, M, s_delta, a):
    """Calcula dx/dt para todos los tokens, con la misma forma que x (L, D)."""
    P = matriz_P(x, M, s_delta, a)
    return np.einsum("dlj,jd->ld", P, x)


def simular(x0, M, s_delta, a, t_max, dt=1e-3, limite=1e8, cada=1, paso_relativo=None):
    """Integra la EDO de tokens con Runge-Kutta de orden 4.

    Por defecto el paso es fijo e igual a dt. Si se da paso_relativo, el paso
    en cada instante es max(dt, paso_relativo * t), lo que permite llegar a
    tiempos muy grandes con pocos pasos cuando la dinámica es lenta.

    Se detiene si algún valor supera el límite en valor absoluto o deja de ser
    finito, lo que sirve para detectar la explosión en tiempo finito. En ese
    caso el último estado guardado es el último estado finito. Guarda un punto
    de cada "cada" pasos y siempre guarda el estado final.

    Devuelve (t, X, exploto) con t de forma (n,), X de forma (n, L, D) y
    exploto como booleano.
    """
    x = np.array(x0, dtype=float)
    f = lambda y: campo_vectorial(y, M, s_delta, a)

    t = 0.0
    ts, xs = [t], [x.copy()]
    exploto = False
    paso = 0

    with np.errstate(over="ignore", invalid="ignore"):
        while t < t_max:
            h = dt if paso_relativo is None else max(dt, paso_relativo * t)
            h = min(h, t_max - t)
            k1 = f(x)
            k2 = f(x + 0.5 * h * k1)
            k3 = f(x + 0.5 * h * k2)
            k4 = f(x + h * k3)
            x_nuevo = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            if not np.all(np.isfinite(x_nuevo)) or np.max(np.abs(x_nuevo)) > limite:
                exploto = True
                break

            x = x_nuevo
            t += h
            paso += 1
            if paso % cada == 0:
                ts.append(t)
                xs.append(x.copy())

    if ts[-1] != t:
        ts.append(t)
        xs.append(x.copy())

    return np.array(ts), np.array(xs), exploto


def norma_por_posicion(X):
    """Calcula la norma de cada token en cada instante, forma (n, L)."""
    return np.linalg.norm(X, axis=2)


def clasificar_1d(mu, s_delta, x0):
    """Clasifica el escenario del caso unidimensional según el signo de mu.

    Devuelve "convergencia" si mu < 0, "divergencia_lenta" si mu > 0 y
    s_delta * x_{l0} < 0 para todo l, y "divergencia_rapida" si mu > 0 y
    s_delta * x_{l0} > 0 para algún l. Supone x0 sin ceros.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    if mu < 0:
        return "convergencia"
    if np.all(s_delta * x0 < 0):
        return "divergencia_lenta"
    return "divergencia_rapida"


def simular_1d(x0, mu, s_delta, a, t_max, **kwargs):
    """Atajo para el caso D = 1 con parámetros escalares.

    x0 es un vector de L tokens. Devuelve lo mismo que simular.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1, 1)
    M = np.array([[mu]], dtype=float)
    s = np.array([[s_delta]], dtype=float)
    return simular(x0, M, s, np.array([a], dtype=float), t_max, **kwargs)


if __name__ == "__main__":
    x0 = np.array([1.0, -0.7, 0.5, 0.9])
    casos = {
        "convergencia (mu < 0)": dict(mu=-1.0, s_delta=1.0),
        "divergencia lenta (mu > 0, s_delta x0 < -2.12 y decreciente)": dict(
            mu=1.0, s_delta=-3.0, x0=np.array([0.8, 0.9, 1.0, 1.1])),
        "divergencia rapida (mu > 0, s_delta x0 > 0)": dict(mu=1.0, s_delta=1.0, x0=np.abs(x0)),
    }
    for nombre, c in casos.items():
        xi = c.pop("x0", x0)
        escenario = clasificar_1d(c["mu"], c["s_delta"], xi)
        t, X, exploto = simular_1d(xi, a=1.0, t_max=20.0, dt=1e-3, cada=100, **c)
        print(f"{nombre}: {escenario}, t final = {t[-1]:.3f}, "
              f"max |x| final = {np.max(np.abs(X[-1])):.3g}, exploto = {exploto}")
