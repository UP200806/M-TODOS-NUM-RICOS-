# M-TODOS-NUM-RICOS-
 Método de Newton-Raphson
import numpy as np

def newton_raphson_ode(f, df, y0, t_span, h):
    """
    Resuelve una ecuación diferencial ordinaria (ODE) utilizando el método de Newton-Raphson.

    Parámetros:
    f (función): La función que define la ODE, dy/dt = f(t, y).
    df (función): La derivada parcial de f con respecto a y, df/dy.
    y0 (float): Condición inicial, y(t0) = y0.
    t_span (tuple): Intervalo de tiempo (t0, tf) para la solución.
    h (float): Tamaño del paso de tiempo.

    Retorna:
    tuple: Arreglos de tiempos y soluciones correspondientes.
    """
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    n = len(t)
    y = np.zeros(n)
    y[0] = y0

    for i in range(n - 1):
        # Predicción utilizando el método de Euler explícito
        y_pred = y[i] + h * f(t[i], y[i])

        # Corrección utilizando el método de Newton-Raphson
        y_corr = y_pred
        for _ in range(10):  # Iterar hasta la convergencia
            y_corr = y_corr - (y_corr - y[i] - h * f(t[i+1], y_corr)) / (1 - h * df(t[i+1], y_corr))

        y[i+1] = y_corr

    return t, y

# Ejemplo de uso:
if __name__ == '__main__':
    # Definir la ODE: dy/dt = -y
    def f(t, y):
        return -y

    # Definir la derivada parcial de f con respecto a y: df/dy = -1
    def df(t, y):
        return -1

    # Condición inicial
    y0 = 1.0

    # Intervalo de tiempo
    t_span = (0, 5)

    # Tamaño del paso de tiempo
    h = 0.1

    # Resolver la ODE
    t, y = newton_raphson_ode(f, df, y0, t_span, h)

    # Imprimir los resultados
    for time, solution in zip(t, y):
        print(f"t = {time:.2f}, y = {solution:.4f}")
