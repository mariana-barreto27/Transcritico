"""Modelo auxiliar para validacao contra condicoes experimentais do artigo.

Este arquivo nao altera o modelo principal. Ele reaproveita as funcoes de
propriedades e aplica a mesma logica termodinamica para aquecimento de agua
liquida, que e o caso experimental de White et al. (2002).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from property_functions import FLUID_CO2, FLUID_WATER, PropertyError, celsius, mpa, props, state_ph, state_pt


@dataclass(frozen=True)
class ValidationInputs:
    Q_sink: float = 120e3
    T_feedwater: float = 20.5 + 273.15
    eta_is_comp: float = 0.70
    eta_motor: float = 0.95
    epsilon_IHX: float = 0.70
    deltaT_min_gascooler: float = 5.0
    P_high_min: float = 9e6
    P_high_max: float = 13e6
    pressure_step: float = 0.5e6
    N_segments: int = 300


def hot_water_side(T_water_out: float, inputs: ValidationInputs) -> dict[str, float]:
    """Propriedades da agua liquida aquecida no gas cooler."""
    P_sat_out = props("P", "T", T_water_out, "Q", 0, FLUID_WATER)
    P_water = max(3e5, P_sat_out + 1e5)
    h_in = props("H", "T", inputs.T_feedwater, "P", P_water, FLUID_WATER)
    h_out = props("H", "T", T_water_out, "P", P_water, FLUID_WATER)
    m_water = inputs.Q_sink / (h_out - h_in)
    return {
        "P_water": P_water,
        "h_water_in": h_in,
        "h_water_out": h_out,
        "m_water": m_water,
        "T_water_out": T_water_out,
    }


def low_state(T_evap: float) -> dict:
    """Estado 6: vapor saturado de CO2 na temperatura de evaporacao."""
    P_low = props("P", "T", T_evap, "Q", 1, FLUID_CO2)
    h6 = props("H", "P", P_low, "Q", 1, FLUID_CO2)
    s6 = props("S", "P", P_low, "Q", 1, FLUID_CO2)
    return {"P_low": P_low, "state6": {"P": P_low, "T": T_evap, "h": h6, "s": s6, "x": 1.0}}


def evaluate_hot_water_case(
    P_high: float,
    T3: float,
    T_evap: float,
    T_water_out: float,
    inputs: ValidationInputs,
    water: dict[str, float],
) -> dict:
    """Calcula o ciclo para uma pressao alta e uma tentativa de T3."""
    low = low_state(T_evap)
    P_low = low["P_low"]
    state6 = low["state6"]
    h6 = state6["h"]
    T6 = state6["T"]

    if P_high <= P_low:
        raise PropertyError("Pressao alta menor ou igual a pressao baixa.")
    if T3 <= T6:
        raise PropertyError("T3 menor ou igual a T6.")

    state3 = state_pt(P_high, T3, FLUID_CO2)
    h3 = state3["h"]

    h4_lim = props("H", "P", P_high, "T", T6, FLUID_CO2)
    h1_lim = props("H", "P", P_low, "T", T3, FLUID_CO2)
    qmax_hot = h3 - h4_lim
    qmax_cold = h1_lim - h6
    qmax = min(qmax_hot, qmax_cold)
    if qmax <= 0:
        raise PropertyError("IHX sem calor maximo positivo.")

    q_ihx = inputs.epsilon_IHX * qmax
    h1 = h6 + q_ihx
    h4 = h3 - q_ihx
    state1 = state_ph(P_low, h1, FLUID_CO2)
    state4 = state_ph(P_high, h4, FLUID_CO2)

    if T3 - state1["T"] <= 0 or state4["T"] - T6 <= 0:
        raise PropertyError("Cruzamento de temperatura no IHX.")

    h2s = props("H", "P", P_high, "S", state1["s"], FLUID_CO2)
    h2 = h1 + (h2s - h1) / inputs.eta_is_comp
    state2 = state_ph(P_high, h2, FLUID_CO2)

    if state2["T"] <= T_water_out + inputs.deltaT_min_gascooler:
        raise PropertyError("Temperatura de descarga insuficiente para aquecer a agua.")

    h_co2 = np.linspace(h3, h2, inputs.N_segments)
    h_water = np.linspace(water["h_water_in"], water["h_water_out"], inputs.N_segments)
    try:
        T_co2 = np.asarray(PropsSI("T", "P", P_high, "H", h_co2, FLUID_CO2), dtype=float)
        T_water = np.asarray(PropsSI("T", "P", water["P_water"], "H", h_water, FLUID_WATER), dtype=float)
    except Exception as exc:
        raise PropertyError("Falha ao calcular perfil do gas cooler.") from exc
    if not (np.all(np.isfinite(T_co2)) and np.all(np.isfinite(T_water))):
        raise PropertyError("Perfil do gas cooler contem temperaturas nao finitas.")

    deltaT = T_co2 - T_water
    min_idx = int(np.argmin(deltaT))
    min_approach = float(deltaT[min_idx])
    pinch_location = float(min_idx / (inputs.N_segments - 1))

    state5 = state_ph(P_low, h4, FLUID_CO2)
    q_evap = h6 - h4
    q_gc = h2 - h3
    w_comp = h2 - h1
    if q_evap <= 0 or q_gc <= 0 or w_comp <= 0:
        raise PropertyError("Calores ou trabalho especificos nao positivos.")

    m_co2 = inputs.Q_sink / q_gc
    W_shaft = m_co2 * w_comp
    W_electric = W_shaft / inputs.eta_motor
    COP = inputs.Q_sink / W_electric

    return {
        "valid": True,
        "P_high": P_high,
        "P_low": P_low,
        "T_evap": T_evap,
        "T_water_out": T_water_out,
        "states": {1: state1, 2: state2, 3: state3, 4: state4, 5: state5, 6: state6},
        "q_gc": q_gc,
        "q_evap": q_evap,
        "q_ihx": q_ihx,
        "m_co2": m_co2,
        "m_water": water["m_water"],
        "W_electric": W_electric,
        "COP": COP,
        "gas_cooler_min_approach": min_approach,
        "pinch_location": pinch_location,
        "profiles": {
            "f": np.linspace(0.0, 1.0, inputs.N_segments),
            "T_co2_gc": T_co2,
            "T_water_gc": T_water,
            "deltaT_gc": deltaT,
        },
    }


def solve_hot_water_pressure(
    P_high: float,
    T_evap: float,
    T_water_out: float,
    inputs: ValidationInputs,
) -> dict:
    """Resolve T3 para que o gas cooler tenha pinch minimo especificado."""
    Pcrit = PropsSI("PCRIT", FLUID_CO2)
    if P_high <= Pcrit:
        return invalid_row(P_high, T_evap, T_water_out, "Pressao alta abaixo da pressao critica do CO2.")

    water = hot_water_side(T_water_out, inputs)

    def residual(T3: float) -> float:
        result = evaluate_hot_water_case(P_high, T3, T_evap, T_water_out, inputs, water)
        return result["gas_cooler_min_approach"] - inputs.deltaT_min_gascooler

    T_low = max(T_evap + 0.5, inputs.T_feedwater + 0.5)
    T_high = min(T_water_out + 120.0, 520.0)
    grid = np.linspace(T_low, T_high, 30)

    samples: list[tuple[float, float]] = []
    for T in grid:
        try:
            samples.append((float(T), float(residual(float(T)))))
        except Exception:
            pass

    bracket: Optional[tuple[float, float]] = None
    for (Ta, ra), (Tb, rb) in zip(samples, samples[1:]):
        if abs(ra) < 1e-8:
            bracket = (Ta, Ta)
            break
        if ra * rb < 0:
            bracket = (Ta, Tb)
            break
    if bracket is None:
        return invalid_row(P_high, T_evap, T_water_out, "Nao foi encontrada solucao de T3 para pinch.")

    try:
        T3 = bracket[0] if bracket[0] == bracket[1] else brentq(residual, bracket[0], bracket[1])
        return evaluate_hot_water_case(P_high, T3, T_evap, T_water_out, inputs, water)
    except Exception as exc:
        return invalid_row(P_high, T_evap, T_water_out, str(exc))


def invalid_row(P_high: float, T_evap: float, T_water_out: float, reason: str) -> dict:
    return {
        "valid": False,
        "P_high": P_high,
        "T_evap": T_evap,
        "T_water_out": T_water_out,
        "invalid_reason": reason,
    }


def row_from_result(result: dict, case_id: str) -> dict:
    row = {
        "case_id": case_id,
        "T_water_out_C": celsius(result["T_water_out"]),
        "T_evap_C": celsius(result["T_evap"]),
        "P_high_MPa": mpa(result["P_high"]),
        "valid": result.get("valid", False),
        "invalid_reason": result.get("invalid_reason", ""),
    }
    if not result.get("valid", False):
        return row

    states = result["states"]
    row.update(
        {
            "COP_model": result["COP"],
            "Q_sink_kW_assumed": 120.0,
            "W_electric_kW": result["W_electric"] / 1e3,
            "m_CO2_kg_s": result["m_co2"],
            "m_water_kg_s": result["m_water"],
            "T1_C": celsius(states[1]["T"]),
            "T2_C": celsius(states[2]["T"]),
            "T3_C": celsius(states[3]["T"]),
            "T4_C": celsius(states[4]["T"]),
            "gas_cooler_min_approach_K": result["gas_cooler_min_approach"],
            "pinch_location": result["pinch_location"],
        }
    )
    return row


def run_validation_sweep(inputs: ValidationInputs) -> pd.DataFrame:
    """Roda as condicoes experimentais principais do artigo."""
    water_out_values = [65.0, 77.5, 90.0]
    evap_values_by_water_out = {
        65.0: [-6.4],
        77.5: [-6.4, 0.3],
        90.0: [-6.4, -3.0, 0.3],
    }
    pressures = np.arange(inputs.P_high_min, inputs.P_high_max + 0.5 * inputs.pressure_step, inputs.pressure_step)

    rows = []
    for T_water_out_C in water_out_values:
        for T_evap_C in evap_values_by_water_out[T_water_out_C]:
            case_id = f"HW{str(T_water_out_C).replace('.', 'p')}_Tevap{str(T_evap_C).replace('-', 'm').replace('.', 'p')}"
            for P_high in pressures:
                result = solve_hot_water_pressure(
                    float(P_high),
                    T_evap_C + 273.15,
                    T_water_out_C + 273.15,
                    inputs,
                )
                rows.append(row_from_result(result, case_id))
    return pd.DataFrame(rows)


def optimum_by_case(sweep: pd.DataFrame) -> pd.DataFrame:
    valid = sweep[sweep["valid"] == True].copy()  # noqa: E712
    rows = []
    for case_id, group in valid.groupby("case_id"):
        best = group.loc[group["COP_model"].idxmax()].copy()
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)
