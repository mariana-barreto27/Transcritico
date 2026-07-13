"""Perfis de temperatura para os trocadores de calor."""

from __future__ import annotations

import numpy as np

from property_functions import celsius


def gas_cooler_profile(result: dict) -> dict[str, np.ndarray | float]:
    """Perfil T-Q normalizado do gas cooler para um resultado valido."""
    f = result["profiles"]["f"]
    return {
        "Q_fraction": f,
        "T_CO2_C": np.array([celsius(T) for T in result["profiles"]["T_co2_gc"]]),
        "T_water_C": np.array([celsius(T) for T in result["profiles"]["T_water_gc"]]),
        "pinch_location": result["pinch_location"],
        "min_deltaT_K": result["gas_cooler_min_approach"],
    }


def ihx_profile(result: dict, n_points: int = 80) -> dict[str, np.ndarray]:
    """Perfil aproximado do IHX, usando variacao linear de entalpia nos dois lados."""
    states = result["states"]
    f = np.linspace(0.0, 1.0, n_points)
    T_hot = np.linspace(states[3]["T"], states[4]["T"], n_points)
    T_cold = np.linspace(states[1]["T"], states[6]["T"], n_points)
    return {
        "Q_fraction": f,
        "T_hot_C": np.array([celsius(T) for T in T_hot]),
        "T_cold_C": np.array([celsius(T) for T in T_cold]),
    }
