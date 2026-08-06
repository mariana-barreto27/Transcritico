
"""Geracao dos graficos do modelo."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

from heat_exchangers import gas_cooler_profile, ihx_profile
from property_functions import FLUID_CO2, kjkg, kjkgk, mpa


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["valid"] == True].copy()  # noqa: E712


def plot_pressure_curves(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = _valid(df)

    plots = [
        ("COP", "COP [-]", "cop_vs_pressao.png"),
        ("T2_C", "Temperatura de descarga [degC]", "temperatura_descarga_vs_pressao.png"),
        ("m_CO2_kg_s", "Vazao de CO2 [kg/s]", "vazao_co2_vs_pressao.png"),
        ("T3_C", "Temperatura de saida do gas cooler [degC]", "t3_vs_pressao.png"),
    ]
    for column, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(valid["P_high_MPa"], valid[column], marker="o", linewidth=1.6)
        ax.set_xlabel("Pressao alta [MPa]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def plot_gas_cooler_TQ(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = gas_cooler_profile(result)
    q_fraction = np.asarray(profile["Q_fraction"], dtype=float)
    t_co2 = np.asarray(profile["T_CO2_C"], dtype=float)
    t_water = np.asarray(profile["T_water_C"], dtype=float)
    delta_t = t_co2 - t_water
    idx = int(np.argmin(delta_t))
    pinch = float(q_fraction[idx])
    t_co2_pinch = float(t_co2[idx])
    t_water_pinch = float(t_water[idx])
    min_delta_t = float(delta_t[idx])
    p_high_mpa = float(result["P_high"]) / 1e6
    cop = float(result.get("COP", np.nan))

    co2_color = "#0f6fb8"
    water_color = "#e76f2e"
    pinch_color = "#c62828"
    fill_color = "#9ecae1"
    grid_color = "#d3d7dc"

    fig, (ax, ax_dt) = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.0], "hspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    ax.set_title(
        f"Perfil T-Q do gas cooler | P alta = {p_high_mpa:.2f} MPa | COP = {cop:.2f}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.fill_between(q_fraction, t_water, t_co2, color=fill_color, alpha=0.18, label="Diferenca de temperatura")
    ax.plot(q_fraction, t_co2, color=co2_color, label="CO$_2$", linewidth=2.7, solid_capstyle="round")
    ax.plot(q_fraction, t_water, color=water_color, label="Agua/vapor", linewidth=2.7, solid_capstyle="round")
    ax.vlines(pinch, t_water_pinch, t_co2_pinch, color=pinch_color, linewidth=1.5)
    ax.scatter([pinch, pinch], [t_co2_pinch, t_water_pinch], color=pinch_color, s=42, zorder=5)
    ax.annotate(
        "",
        xy=(pinch, t_co2_pinch),
        xytext=(pinch, t_water_pinch),
        arrowprops={"arrowstyle": "<->", "color": pinch_color, "linewidth": 1.5},
    )
    ax.annotate(
        f"Pinch\n$\\Delta T_{{min}}$ = {min_delta_t:.2f} K",
        xy=(pinch, 0.5 * (t_co2_pinch + t_water_pinch)),
        xycoords="data",
        xytext=(0.62, 0.38),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=10,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.96},
        arrowprops={"arrowstyle": "->", "color": pinch_color, "linewidth": 1.2},
    )
    ax.set_ylabel("Temperatura [degC]")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#dddddd", framealpha=0.96)

    ax_dt.plot(q_fraction, delta_t, color="#4b5563", linewidth=2.0)
    ax_dt.axhline(profile["min_deltaT_K"], color=pinch_color, linestyle="--", linewidth=1.3)
    ax_dt.scatter([pinch], [min_delta_t], color=pinch_color, s=36, zorder=5)
    ax_dt.annotate(
        f"{min_delta_t:.2f} K",
        xy=(pinch, min_delta_t),
        xytext=(8, 9),
        textcoords="offset points",
        color=pinch_color,
        fontsize=9,
        fontweight="bold",
    )
    ax_dt.set_ylabel("$\\Delta T$ [K]")
    ax_dt.set_xlabel("Calor transferido acumulado, $Q/Q_{total}$")
    ax_dt.set_xlim(0.0, 1.0)
    ax_dt.set_xticks(np.linspace(0.0, 1.0, 6))
    ax_dt.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    for axis in (ax, ax_dt):
        axis.set_facecolor("#fbfbfa")
        axis.grid(True, color=grid_color, linewidth=0.8, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#9aa0a6")
        axis.spines["bottom"].set_color("#9aa0a6")

    fig.subplots_adjust(top=0.90, hspace=0.08)
    fig.savefig(output_dir / "tq_gas_cooler_otimo.png", dpi=220)
    plt.close(fig)


def plot_ihx_profile(result: dict, output_dir: Path) -> None:
    profile = ihx_profile(result)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(profile["Q_fraction"], profile["T_hot_C"], label="Lado quente: 3 -> 4", linewidth=2)
    ax.plot(profile["Q_fraction"], profile["T_cold_C"], label="Lado frio: 1 -> 6", linewidth=2)
    ax.set_xlabel("Fracao do calor transferido [-]")
    ax.set_ylabel("Temperatura [degC]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "perfil_ihx_otimo.png", dpi=160)
    plt.close(fig)


def _valid_results(results: list[dict]) -> list[dict]:
    return [r for r in results if r.get("valid", False)]


def _extreme_pressure_results(results: list[dict]) -> list[dict]:
    valid = sorted(_valid_results(results), key=lambda r: r["P_high"])
    if not valid:
        return []
    if len(valid) == 1:
        return [valid[0]]
    return [valid[0], valid[-1]]


def _co2_saturation_dome() -> dict[str, np.ndarray]:
    T_triple = PropsSI("TTRIPLE", FLUID_CO2)
    T_crit = PropsSI("TCRIT", FLUID_CO2)
    temperatures = np.linspace(T_triple + 0.2, T_crit - 0.05, 260)
    sat = {"P_MPa": [], "h_liq": [], "h_vap": [], "s_liq": [], "s_vap": []}
    for T in temperatures:
        try:
            P = PropsSI("P", "T", T, "Q", 0, FLUID_CO2)
            sat["P_MPa"].append(mpa(P))
            sat["h_liq"].append(kjkg(PropsSI("H", "T", T, "Q", 0, FLUID_CO2)))
            sat["h_vap"].append(kjkg(PropsSI("H", "T", T, "Q", 1, FLUID_CO2)))
            sat["s_liq"].append(kjkgk(PropsSI("S", "T", T, "Q", 0, FLUID_CO2)))
            sat["s_vap"].append(kjkgk(PropsSI("S", "T", T, "Q", 1, FLUID_CO2)))
        except Exception:
            continue
    return {key: np.asarray(value, dtype=float) for key, value in sat.items()}


def _cycle_arrays(result: dict) -> dict[str, list[float]]:
    order = [1, 2, 3, 4, 5, 6, 1]
    states = result["states"]
    return {
        "order": order,
        "P_MPa": [mpa(states[i]["P"]) for i in order],
        "h_kJ_kg": [kjkg(states[i]["h"]) for i in order],
        "s_kJ_kgK": [kjkgk(states[i]["s"]) for i in order],
    }


def _annotate_cycle_states(
    ax: plt.Axes,
    cycle: dict[str, list[float]],
    x_key: str,
    offsets: dict[int, tuple[int, int]],
    color: str,
) -> None:
    for state, x_value, pressure in zip(cycle["order"][:-1], cycle[x_key][:-1], cycle["P_MPa"][:-1]):
        dx, dy = offsets.get(state, (6, 6))
        ax.annotate(
            str(state),
            (x_value, pressure),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "circle,pad=0.22",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.9,
                "alpha": 0.96,
            },
        )


def _finish_phase_axis(ax: plt.Axes, title: str, xlabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_yscale("log")
    ax.set_facecolor("#fbfbfa")
    ax.grid(True, which="major", color="#cfd4da", linewidth=0.8, alpha=0.7)
    ax.grid(True, which="minor", color="#d9dde2", linewidth=0.5, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa0a6")
    ax.spines["bottom"].set_color("#9aa0a6")
    ax.margins(x=0.06, y=0.14)


def _cycle_segment_styles() -> list[tuple[int, int, str, str, str]]:
    return [
        (1, 2, "Compressor", "#b3261e", "-"),
        (2, 3, "Gas cooler", "#0f6fb8", "-"),
        (3, 4, "IHX - lado quente", "#7b3fb2", "-"),
        (4, 5, "Valvula", "#5f6368", ":"),
        (5, 6, "Evaporador", "#188038", "-"),
        (6, 1, "IHX - lado frio", "#d97706", "-"),
    ]


def _draw_cycle_segment(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str, label: str, style: str) -> None:
    ax.plot([x0, x1], [y0, y1], color=color, linestyle=style, linewidth=2.4, label=label, solid_capstyle="round")
    f0, f1 = 0.43, 0.58
    xa = x0 + (x1 - x0) * f0
    xb = x0 + (x1 - x0) * f1
    ya = 10 ** (np.log10(y0) + (np.log10(y1) - np.log10(y0)) * f0)
    yb = 10 ** (np.log10(y0) + (np.log10(y1) - np.log10(y0)) * f1)
    ax.annotate(
        "",
        xy=(xb, yb),
        xytext=(xa, ya),
        arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.4, "mutation_scale": 10},
    )


def _plot_phase_cycle_panel(
    ax: plt.Axes,
    result: dict,
    dome: dict[str, np.ndarray],
    x_key: str,
    dome_liq_key: str,
    dome_vap_key: str,
    xlabel: str,
    title: str,
    state_offsets: dict[int, tuple[int, int]],
    show_ylabel: bool,
) -> None:
    ax.fill_betweenx(dome["P_MPa"], dome[dome_liq_key], dome[dome_vap_key], color="#9aa0a6", alpha=0.08)
    ax.plot(dome[dome_liq_key], dome["P_MPa"], color="#7a7f85", linewidth=1.15, label="Saturacao CO$_2$")
    ax.plot(dome[dome_vap_key], dome["P_MPa"], color="#7a7f85", linewidth=1.15)

    cycle = _cycle_arrays(result)
    points = {
        state: (cycle[x_key][idx], cycle["P_MPa"][idx])
        for idx, state in enumerate(cycle["order"][:-1])
    }
    for start, end, label, color, style in _cycle_segment_styles():
        x0, y0 = points[start]
        x1, y1 = points[end]
        _draw_cycle_segment(ax, x0, y0, x1, y1, color, label, style)

    for state, (x_value, pressure) in points.items():
        ax.scatter([x_value], [pressure], s=42, facecolor="white", edgecolor="#222222", linewidth=1.1, zorder=5)
    _annotate_cycle_states(ax, cycle, x_key, state_offsets, "#222222")

    _finish_phase_axis(ax, title, xlabel)
    if show_ylabel:
        ax.set_ylabel("Pressao [MPa]")
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.text(
        0.03,
        0.04,
        f"P alta = {mpa(result['P_high']):.2f} MPa\nP baixa = {mpa(result['P_low']):.2f} MPa\nCOP = {result['COP']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.95},
    )


def plot_ph_ps_extreme_pressures(results: list[dict], output_dir: Path) -> None:
    """Gera diagramas P-h e P-s para a menor e maior pressao alta validas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _extreme_pressure_results(results)
    if not selected:
        return

    dome = _co2_saturation_dome()

    ph_offsets = {1: (0, -17), 2: (12, 8), 3: (0, 16), 4: (0, 16), 5: (0, -17), 6: (0, 16)}
    ps_offsets = {1: (0, -17), 2: (12, 8), 3: (0, 16), 4: (0, 16), 5: (0, -17), 6: (0, 16)}
    panel_titles = ["Menor pressao alta valida", "Maior pressao alta valida"]

    fig, axes = plt.subplots(1, len(selected), figsize=(13.2, 5.9), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    for idx, (ax, result) in enumerate(zip(axes, selected)):
        _plot_phase_cycle_panel(
            ax,
            result,
            dome,
            "h_kJ_kg",
            "h_liq",
            "h_vap",
            "Entalpia [kJ/kg]",
            panel_titles[idx],
            ph_offsets,
            idx == 0,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Diagrama P-h do CO$_2$ com componentes do ciclo", fontsize=13, fontweight="bold", y=0.98)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, facecolor="white", edgecolor="#dddddd")
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    fig.savefig(output_dir / "diagrama_ph_extremos_pressao.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(selected), figsize=(13.2, 5.9), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    for idx, (ax, result) in enumerate(zip(axes, selected)):
        _plot_phase_cycle_panel(
            ax,
            result,
            dome,
            "s_kJ_kgK",
            "s_liq",
            "s_vap",
            "Entropia [kJ/(kg K)]",
            panel_titles[idx],
            ps_offsets,
            idx == 0,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Diagrama P-s do CO$_2$ com componentes do ciclo", fontsize=13, fontweight="bold", y=0.98)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, facecolor="white", edgecolor="#dddddd")
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    fig.savefig(output_dir / "diagrama_ps_extremos_pressao.png", dpi=220)
    plt.close(fig)


def plot_ph_ps_optimum(result: dict, output_dir: Path) -> None:
    """Gera diagramas P-h e P-s apenas para o ponto otimo do ciclo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dome = _co2_saturation_dome()
    state_offsets = {1: (0, -17), 2: (12, 8), 3: (0, 16), 4: (0, 16), 5: (0, -17), 6: (0, 16)}

    fig, ax = plt.subplots(figsize=(8.0, 5.9))
    fig.patch.set_facecolor("white")
    _plot_phase_cycle_panel(
        ax,
        result,
        dome,
        "h_kJ_kg",
        "h_liq",
        "h_vap",
        "Entalpia [kJ/kg]",
        "Ponto otimo",
        state_offsets,
        True,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Diagrama P-h do CO$_2$ no ponto otimo", fontsize=13, fontweight="bold", y=0.98)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, facecolor="white", edgecolor="#dddddd")
    fig.tight_layout(rect=(0, 0.18, 1, 0.93))
    fig.savefig(output_dir / "diagrama_ph_otimo.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.9))
    fig.patch.set_facecolor("white")
    _plot_phase_cycle_panel(
        ax,
        result,
        dome,
        "s_kJ_kgK",
        "s_liq",
        "s_vap",
        "Entropia [kJ/(kg K)]",
        "Ponto otimo",
        state_offsets,
        True,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Diagrama P-s do CO$_2$ no ponto otimo", fontsize=13, fontweight="bold", y=0.98)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, facecolor="white", edgecolor="#dddddd")
    fig.tight_layout(rect=(0, 0.18, 1, 0.93))
    fig.savefig(output_dir / "diagrama_ps_otimo.png", dpi=220)
    plt.close(fig)


def plot_sensitivity(curves: dict[float, pd.DataFrame], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps, df in curves.items():
        valid = _valid(df)
        ax.plot(valid["P_high_MPa"], valid["COP"], marker="o", linewidth=1.3, label=f"epsilon={eps:.2f}")
    ax.set_xlabel("Pressao alta [MPa]")
    ax.set_ylabel("COP [-]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sensibilidade_ihx_cop_vs_pressao.png", dpi=160)
    plt.close(fig)
