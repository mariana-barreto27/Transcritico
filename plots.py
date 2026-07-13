"""Geracao dos graficos do modelo."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from heat_exchangers import gas_cooler_profile, ihx_profile


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
    profile = gas_cooler_profile(result)
    pinch = profile["pinch_location"]
    idx = int(round(pinch * (len(profile["Q_fraction"]) - 1)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(profile["Q_fraction"], profile["T_CO2_C"], label="CO2", linewidth=2)
    ax.plot(profile["Q_fraction"], profile["T_water_C"], label="Agua/vapor", linewidth=2)
    ax.scatter([pinch], [profile["T_CO2_C"][idx]], color="tab:red", zorder=5, label="Pinch")
    ax.annotate(
        f"min DeltaT = {profile['min_deltaT_K']:.2f} K",
        xy=(pinch, profile["T_CO2_C"][idx]),
        xytext=(min(0.75, pinch + 0.08), profile["T_CO2_C"][idx] + 10),
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
    )
    ax.set_xlabel("Fracao do calor transferido [-]")
    ax.set_ylabel("Temperatura [degC]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tq_gas_cooler_otimo.png", dpi=160)
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
