#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch

# ----------------------------
# Core math (matches your code)
# ----------------------------
def compute_theta(F_max: float, f_star: float, s_min: float) -> float:
    s_star = f_star / F_max
    if not (s_min < s_star < 1.0):
        raise ValueError("Require s_min < f_star/F_max < 1.0")
    ratio = (s_star - s_min) / (1.0 - s_min)
    return -f_star / np.log(ratio)

def exp_scale(f: float, theta: float, s_min: float) -> float:
    return s_min + (1.0 - s_min) * np.exp(-f / theta)

def g_map(f: float, F_max: float, theta: float, s_min: float) -> float:
    # f_{k+1} = F_max * s(f_k)
    s = exp_scale(f, theta, s_min)
    return F_max * np.clip(s, s_min, 1.0)

# ----------------------------
# Plot helpers
# ----------------------------
def cobweb(ax, g, x0, n_steps, x_min, x_max, color="red", lw=1.2, ls="--", alpha=0.9):
    """
    Standard cobweb for x_{k+1} = g(x_k):
      vertical: (x_k, x_k) -> (x_k, g(x_k))
      horizontal: (x_k, g(x_k)) -> (g(x_k), g(x_k))
    """
    x = float(x0)
    for i in range(n_steps):
        y = float(g(x))
        if i >= 1:
            ax.plot([x, x], [x, y], color=color, lw=lw, ls=ls, alpha=alpha, zorder=3)
        ax.plot([x, y], [y, y], color=color, lw=lw, ls=ls, alpha=alpha, zorder=3)
        x = y

def make_figure(
    F_max=7.0,
    f_star=2.0,
    s_min=0.2,
    n_cycles=30,
    contact_cycle=5,
    f_cmd_pre=5.0,
    f_cmd_post=2.0,
    f_contact_init=5.0,     # initial measured force at the first contact cycle
    cobweb_steps=12,
    grid_points=800,
):
    theta = compute_theta(F_max=F_max, f_star=f_star, s_min=s_min)

    # ---------- Left panel: ideal time response ----------
    cycles = np.arange(n_cycles, dtype=int)

    F_cmd = np.full(n_cycles, f_cmd_post, dtype=float)
    F_cmd[:contact_cycle+1] = f_cmd_pre

    F_meas = np.zeros(n_cycles, dtype=float)
    F_lim  = np.zeros(n_cycles, dtype=float)

    # Before contact: no contact force measured; limiter effectively "high"
    F_meas[:contact_cycle] = 0.0
    F_lim[:contact_cycle] = F_max

    # At first contact, we "hit" with some measured force (gives overshoot in the plot)
    F_meas[contact_cycle] = f_contact_init
    F_lim[contact_cycle]  = g_map(F_meas[contact_cycle], F_max, theta, s_min)

    # After contact: iterate the 1D recursion (idealized closed-loop)
    # Here we model the measured force evolving by the same recurrence; that’s what the cobweb visualizes.
    for k in range(contact_cycle, n_cycles - 1):
        F_meas[k + 1] = g_map(F_meas[k], F_max, theta, s_min)
        F_lim[k + 1]  = g_map(F_meas[k + 1], F_max, theta, s_min)

    # Fine grid for a smoother-looking left plot (optional)
    t_grid = np.linspace(cycles.min(), cycles.max(), grid_points)
    cmd_grid  = np.interp(t_grid, cycles, F_cmd)
    meas_grid = np.interp(t_grid, cycles, F_meas)
    lim_grid  = np.interp(t_grid, cycles, F_lim)

    # ---------- Right panel: phase-plane curve + cobweb ----------
    x_grid = np.linspace(0.0, F_max, grid_points)
    y_grid = np.array([g_map(x, F_max, theta, s_min) for x in x_grid], dtype=float)

    # ---------- Figure layout ----------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": True,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.45,
        "grid.linewidth": 0.4,
        "text.latex.preamble": r"\usepackage{bm}"
    })

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.8, 3.0), sharey=True)

    # ----- Left panel styling -----
    axL.set_xlim(0, n_cycles - 1)
    axL.set_ylim(-1.0, 7.3)
    axL.set_xlabel(r"$\mathbf{Controller\ Cycles}$")
    axL.set_ylabel(r"$\mathbf{Force\ [N]}$")

    # "Before Contact" shaded region
    axL.axvspan(0, contact_cycle, color="0.85", zorder=0)

    # Hatched forbidden region: above adaptive limit
    y_top = axL.get_ylim()[1]
    xs = np.concatenate([t_grid, t_grid[::-1]])
    ys = np.concatenate([np.full_like(t_grid, y_top), lim_grid[::-1]])
    poly = Polygon(
        np.c_[xs, ys],
        closed=True,
        facecolor="none",
        edgecolor="red",
        hatch="///",
        linewidth=0.0,
        alpha=0.35,
        zorder=1
    )
    axL.add_patch(poly)

    # Curves
    axL.plot(t_grid, cmd_grid,  color="blue",  lw=2.0, label=r"$F_{\mathrm{cmd}}$", zorder=4)
    axL.plot(t_grid, meas_grid, color="green", lw=2.0, label=r"$F_{\mathrm{meas}}$", zorder=5)
    axL.plot(t_grid, lim_grid,  color="red",   lw=2.2, label=r"$F_{\mathrm{lim}}(F_{\mathrm{meas}})$", zorder=6)

    # Fixed point line
    axL.axhline(f_star, color="red", lw=2.0, ls=":", label=r"$F_{\mathrm{lim}}^{*}$", zorder=3)

    # Legend entries for patches
    patch_before = Patch(facecolor="0.85", edgecolor="0.85", label="Before Contact")
    axL.legend(handles=axL.get_legend_handles_labels()[0] + [patch_before],
               labels=axL.get_legend_handles_labels()[1] + ["Before Contact"],
               loc="upper right", frameon=True, framealpha=0.9)

    # ----- Right panel styling -----
    axR.set_xlim(0, F_max)
    axR.set_xlabel(r"$F_{\mathrm{meas}}\ \mathrm{[N]}$")

    # g(x) curve and y=x
    axR.plot(x_grid, y_grid, color="red", lw=2.4, label=r"$F_{\mathrm{lim}}(F_{\mathrm{meas}})$", zorder=2)
    axR.plot([0, 7], [0, 7], color="k", lw=1.5, ls="--", alpha=0.8, label=r"$y=x$", zorder=1)

    # Fixed point marker
    axR.plot([f_star], [f_star], "o", color="red", ms=6, label=r"$F_{\mathrm{lim}}^{*}$", zorder=4)

    # Cobweb from the “contact init” value
    g = lambda x: g_map(x, F_max, theta, s_min)
    cobweb(axR, g=g, x0=f_contact_init, n_steps=cobweb_steps, x_min=0.0, x_max=F_max,
           color="red", lw=1.2, ls="--", alpha=0.85)

    axR.legend(loc="upper right", frameon=True, framealpha=0.9)

    for ax in (axL, axR):
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.grid(True)

    fig.tight_layout()
    return fig

if __name__ == "__main__":
    fig = make_figure(
        F_max=7.0,
        f_star=2.0,
        s_min=0.17,
        n_cycles=30,
        contact_cycle=5,
        f_cmd_pre=5.0,
        f_cmd_post=2.0,
        f_contact_init=5.0,
        cobweb_steps=15,
    )
    fig.savefig("adaptive_force_limits_ideal.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig("adaptive_force_limits_ideal.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.show()
