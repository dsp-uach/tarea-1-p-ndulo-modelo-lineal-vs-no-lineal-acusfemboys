#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_compare_models.py

Comparación entre:
- péndulo NO lineal
- péndulo lineal (aproximación)

Objetivo:
Explorar cuándo la aproximación sin(θ) ≈ θ es válida
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros físicos
g = 9.81      # gravedad (m/s^2)
L = 1.0       # longitud (m)

# Condiciones iniciales
theta0_deg = 5  # ángulo inicial en grados
theta0 = np.deg2rad(theta0_deg)  # convertir a radianes
omega0 = 0.0

# Tiempo de simulación
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# Modelo NO lineal
def pendulum_nonlinear(t, y):
    theta, omega = y
    return [omega, -(g/L) * np.sin(theta)]

# Modelo lineal
def pendulum_linear(t, y):
    theta, omega = y
    return [omega, -(g/L) * theta]

# Resolver ecuaciones
def solve_models():
    sol_nl = solve_ivp(
        pendulum_nonlinear,
        t_span,
        [theta0, omega0],
        t_eval=t_eval
    )

    sol_lin = solve_ivp(
        pendulum_linear,
        t_span,
        [theta0, omega0],
        t_eval=t_eval
    )

    return sol_nl, sol_lin

# Gráficos
def plot_results(sol_nl, sol_lin):

    # Crear carpeta de salida
    base_dir = Path(__file__).resolve().parent.parent
    plot_dir = base_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Comparación
    plt.figure()
    plt.plot(sol_nl.t, sol_nl.y[0], label="No lineal")
    plt.plot(sol_lin.t, sol_lin.y[0], "--", label="Lineal")

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (rad)")
    plt.title(f"Comparación (θ0 = {theta0_deg}°)")
    plt.legend()
    plt.grid()

    plt.savefig(plot_dir / "comparison.png", dpi=150)
    plt.show()

    # Error 
    error = sol_nl.y[0] - sol_lin.y[0]

    plt.figure()
    plt.plot(sol_nl.t, error)

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Error (rad)")
    plt.title("Error entre modelo lineal y no lineal")
    plt.grid()

    plt.savefig(plot_dir / "error.png", dpi=150)
    plt.show()

    # Métrica
    max_error = np.max(np.abs(error))
    print("\n=======================")
    print("RESULTADOS")
    print("=======================")
    print(f"Ángulo inicial: {theta0_deg}°")
    print(f"Error máximo: {max_error:.6f} rad")

# MAIN
def main():
    print("Resolviendo modelos...")

    sol_nl, sol_lin = solve_models()

    print("Generando gráficos...")
    plot_results(sol_nl, sol_lin)


if __name__ == "__main__":
    main()