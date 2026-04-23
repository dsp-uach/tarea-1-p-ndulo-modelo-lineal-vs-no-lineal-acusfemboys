#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_compare_models_L_sweep.py

Comparación entre péndulo no lineal y lineal variando la LONGITUD L.
Analiza cómo el error máximo depende de L para un tiempo fijo de simulación.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Forzar backend no interactivo si se prefiere solo guardar archivos
import matplotlib
matplotlib.use('Agg')   # Descomenta para evitar ventanas emergentes

# Parámetros físicos fijos
g = 9.81                     # gravedad (m/s^2)
theta0_deg = 40              # ángulo inicial fijo (grados) para apreciar error
theta0 = np.deg2rad(theta0_deg)
omega0 = 0.0

# Tiempo de simulación fijo
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# Valores de L a explorar (metros)
L_values = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# Modelos: ahora reciben L como argumento
def pendulum_nonlinear(t, y, L):
    theta, omega = y
    return [omega, -(g/L) * np.sin(theta)]

def pendulum_linear(t, y, L):
    theta, omega = y
    return [omega, -(g/L) * theta]

def solve_for_L(L):
    """Resuelve ambos modelos para una L dada."""
    sol_nl = solve_ivp(
        lambda t, y: pendulum_nonlinear(t, y, L),
        t_span,
        [theta0, omega0],
        t_eval=t_eval,
        rtol=1e-8, atol=1e-10
    )
    sol_lin = solve_ivp(
        lambda t, y: pendulum_linear(t, y, L),
        t_span,
        [theta0, omega0],
        t_eval=t_eval,
        rtol=1e-8, atol=1e-10
    )
    return sol_nl, sol_lin

def main():
    try:
        # Configurar carpeta de salida
        # Si el script se ejecuta desde cualquier lugar, usamos una ruta relativa segura
        script_dir = Path(__file__).resolve().parent
        # Suponemos estructura: proyecto/
        #                       ├── src/ (o scripts/)
        #                       └── results/plots/
        # Si no existe la carpeta results en el padre, la creamos en el directorio actual
        parent_dir = script_dir.parent
        results_dir = parent_dir / "results"
        if not results_dir.exists():
            # Si no hay results en el padre, usar directorio actual
            results_dir = Path.cwd() / "results"
        plot_dir = results_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Los gráficos se guardarán en: {plot_dir.resolve()}")

        max_errors = []

        print("\n==============================================")
        print(f"BARRIDO DE LONGITUD L (θ0 = {theta0_deg}°)")
        print("==============================================")

        for L in L_values:
            sol_nl, sol_lin = solve_for_L(L)
            error = sol_nl.y[0] - sol_lin.y[0]
            max_err = np.max(np.abs(error))
            max_errors.append(max_err)
            print(f"L = {L:.2f} m  |  Error máximo = {max_err:.6f} rad")

        # Graficar error máximo vs L
        plt.figure(figsize=(8, 5))
        plt.plot(L_values, max_errors, 'o-', linewidth=2, markersize=8)
        plt.xlabel("Longitud del péndulo (m)")
        plt.ylabel("Error máximo (rad)")
        plt.title(f"Error máximo entre modelo no lineal y lineal\nθ0 = {theta0_deg}°, tiempo simulación = 10 s")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        file1 = plot_dir / "max_error_vs_L.png"
        plt.savefig(file1, dpi=150)
        print(f"Guardado: {file1.resolve()}")

        # Mostrar si hay backend interactivo
        if plt.get_backend() != 'agg':
            plt.show()
        plt.close()

        # Gráfico en escala log-log
        plt.figure(figsize=(8, 5))
        plt.loglog(L_values, max_errors, 'o-', linewidth=2, markersize=8)
        plt.xlabel("Longitud (m)")
        plt.ylabel("Error máximo (rad)")
        plt.title("Error máximo vs L (escala log-log)")
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.tight_layout()
        file2 = plot_dir / "max_error_vs_L_loglog.png"
        plt.savefig(file2, dpi=150)
        print(f"Guardado: {file2.resolve()}")

        if plt.get_backend() != 'agg':
            plt.show()
        plt.close()

        # Ajuste para verificar dependencia ~ L^{-0.5}
        logL = np.log(L_values)
        logErr = np.log(max_errors)
        slope, intercept = np.polyfit(logL, logErr, 1)
        print("\nAjuste en escala log-log: log(error) = {:.3f} * log(L) + {:.3f}".format(slope, intercept))
        print(f"Pendiente ≈ {slope:.3f} (esperado -0.5 para tiempo fijo)")

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
