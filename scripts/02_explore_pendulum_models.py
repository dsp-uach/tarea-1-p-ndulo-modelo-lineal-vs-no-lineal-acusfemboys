#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_explore_pendulum_models.py

Explora el comportamiento del péndulo lineal y no lineal mediante:

1. Barrido automático de ángulos iniciales
2. Comparación de períodos
3. Animación simple del péndulo
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# CONFIGURACIÓN
g = 9.81
L = 1.0
omega0 = 0.0

theta0_deg_list = [5, 10, 20, 40, 60]

t_span = (0, 15)
t_eval = np.linspace(*t_span, 3000)

SAVE_FIGURES = True
SAVE_ANIMATION = False   # Cambiar a True si quieres guardar GIF/MP4

ANIMATION_THETA0_DEG = 5

# RUTAS
BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_DIR = BASE_DIR / "results" / "plots"
ANIM_DIR = BASE_DIR / "results" / "animations"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
ANIM_DIR.mkdir(parents=True, exist_ok=True)

# MODELOS
def pendulum_nonlinear(t, y):
    theta, omega = y
    return [omega, -(g / L) * np.sin(theta)]


def pendulum_linear(t, y):
    theta, omega = y
    return [omega, -(g / L) * theta]

# RESOLVER MODELOS
def solve_models(theta0_deg):
    theta0 = np.deg2rad(theta0_deg)

    sol_nl = solve_ivp(
        pendulum_nonlinear,
        t_span,
        [theta0, omega0],
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    sol_lin = solve_ivp(
        pendulum_linear,
        t_span,
        [theta0, omega0],
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    return sol_nl, sol_lin

# ESTIMAR PERÍODO
def estimate_period(time, theta_signal):
    """
    Estima período usando peaks sobre la señal angular.
    Busca máximos positivos.
    """
    peaks, _ = find_peaks(theta_signal, distance=50)

    if len(peaks) < 2:
        return np.nan

    peak_times = time[peaks]
    periods = np.diff(peak_times)

    if len(periods) == 0:
        return np.nan

    return np.mean(periods)

# COMPARACIÓN PARA UN ÁNGULO
def plot_single_comparison(theta0_deg, sol_nl, sol_lin):
    error = sol_nl.y[0] - sol_lin.y[0]

    # Comparación de respuestas
    plt.figure(figsize=(8, 4))
    plt.plot(sol_nl.t, sol_nl.y[0], label="No lineal")
    plt.plot(sol_lin.t, sol_lin.y[0], "--", label="Lineal")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (rad)")
    plt.title(f"Comparación de modelos (θ0 = {theta0_deg}°)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(PLOT_DIR / f"comparison_{theta0_deg}deg.png", dpi=150, bbox_inches="tight")

    plt.show()

    # Error
    plt.figure(figsize=(8, 4))
    plt.plot(sol_nl.t, error)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Error (rad)")
    plt.title(f"Error entre modelos (θ0 = {theta0_deg}°)")
    plt.grid(True)
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(PLOT_DIR / f"error_{theta0_deg}deg.png", dpi=150, bbox_inches="tight")

    plt.show()

# BARRIDO AUTOMÁTICO
def angle_sweep():
    max_errors = []
    T_linear_list = []
    T_nonlinear_list = []

    print("\n==============================")
    print("BARRIDO AUTOMÁTICO DE ÁNGULOS")
    print("==============================")

    for theta0_deg in theta0_deg_list:
        sol_nl, sol_lin = solve_models(theta0_deg)

        error = sol_nl.y[0] - sol_lin.y[0]
        max_error = np.max(np.abs(error))

        T_nl = estimate_period(sol_nl.t, sol_nl.y[0])
        T_lin = estimate_period(sol_lin.t, sol_lin.y[0])

        max_errors.append(max_error)
        T_linear_list.append(T_lin)
        T_nonlinear_list.append(T_nl)

        print(f"\nθ0 = {theta0_deg}°")
        print(f"  Error máximo     = {max_error:.6f} rad")
        print(f"  T lineal         = {T_lin:.6f} s")
        print(f"  T no lineal      = {T_nl:.6f} s")
        if np.isfinite(T_lin) and np.isfinite(T_nl):
            print(f"  ΔT               = {T_nl - T_lin:.6f} s")

        # Mostrar comparación individual
        plot_single_comparison(theta0_deg, sol_nl, sol_lin)

    return np.array(max_errors), np.array(T_linear_list), np.array(T_nonlinear_list)

# GRÁFICOS RESUMEN
def plot_summary(max_errors, T_linear_list, T_nonlinear_list):

    # Error máximo vs ángulo
    plt.figure(figsize=(8, 4))
    plt.plot(theta0_deg_list, max_errors, marker="o")
    plt.xlabel("Ángulo inicial (grados)")
    plt.ylabel("Error máximo (rad)")
    plt.title("Error máximo vs ángulo inicial")
    plt.grid(True)
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(PLOT_DIR / "max_error_vs_angle.png", dpi=150, bbox_inches="tight")

    plt.show()

    # Períodos lineal vs no lineal
    plt.figure(figsize=(8, 4))
    plt.plot(theta0_deg_list, T_linear_list, marker="o", label="Lineal")
    plt.plot(theta0_deg_list, T_nonlinear_list, marker="s", label="No lineal")
    plt.xlabel("Ángulo inicial (grados)")
    plt.ylabel("Período (s)")
    plt.title("Período vs ángulo inicial")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(PLOT_DIR / "period_vs_angle.png", dpi=150, bbox_inches="tight")

    plt.show()

# ANIMACIÓN
def animate_pendulum(theta0_deg):
    sol_nl, sol_lin = solve_models(theta0_deg)

    theta_nl = sol_nl.y[0]
    theta_lin = sol_lin.y[0]
    t = sol_nl.t

    # Coordenadas
    x_nl = L * np.sin(theta_nl)
    y_nl = -L * np.cos(theta_nl)

    x_lin = L * np.sin(theta_lin)
    y_lin = -L * np.cos(theta_lin)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.2 * L, 1.2 * L)
    ax.set_ylim(-1.2 * L, 0.2 * L)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(f"Animación del péndulo (θ0 = {theta0_deg}°)")

    line_nl, = ax.plot([], [], "o-", label="No lineal")
    line_lin, = ax.plot([], [], "o--", label="Lineal")
    time_text = ax.text(-1.1 * L, 0.05 * L, "")

    ax.legend()

    def update(frame):
        line_nl.set_data([0, x_nl[frame]], [0, y_nl[frame]])
        line_lin.set_data([0, x_lin[frame]], [0, y_lin[frame]])
        time_text.set_text(f"t = {t[frame]:.2f} s")
        return line_nl, line_lin, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=10,
        blit=True
    )

    if SAVE_ANIMATION:
        try:
            ani.save(ANIM_DIR / f"pendulum_animation_{theta0_deg}deg.gif", writer="pillow", fps=30)
            print(f"Animación guardada en: {ANIM_DIR}")
        except Exception as e:
            print(f"No se pudo guardar la animación: {e}")

    plt.show()

# MAIN
def main():
    print("Resolviendo barrido de ángulos...")

    max_errors, T_linear_list, T_nonlinear_list = angle_sweep()

    print("\nGenerando gráficos resumen...")
    plot_summary(max_errors, T_linear_list, T_nonlinear_list)

    print(f"\nGenerando animación para θ0 = {ANIMATION_THETA0_DEG}° ...")
    animate_pendulum(ANIMATION_THETA0_DEG)

if __name__ == "__main__":
    main()

L_values = [0.5, 1.0, 1.5, 2.0]   # metros
theta0_fixed = 40  # grados

errors_max_vs_L = []
for L in L_values:
    # Actualizar parámetros globales (g, L, etc.)
    sol_nl, sol_lin = solve_models(theta0_fixed, L)   # modificar funciones para aceptar L
    error = sol_nl.y[0] - sol_lin.y[0]
    errors_max_vs_L.append(np.max(np.abs(error)))