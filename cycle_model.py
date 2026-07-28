"""Modelo de bomba de calor transcritica de CO2 com IHX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq, minimize_scalar

from property_functions import (
    FLUID_CO2,
    FLUID_WATER,
    PropertyError,
    celsius,
    kjkg,
    kjkgk,
    mpa,
    props,
    quality_or_none,
    state_ph,
    state_pt,
)


@dataclass(frozen=True)
class CycleInputs:
    Q_sink: float = 120e3
    T_steam: float = 120.0 + 273.15
    T_feedwater: float = 20.0 + 273.15
    T_evap: float = 20.0 + 273.15
    eta_is_comp: float = 0.70
    eta_motor: float = 0.95
    epsilon_IHX: float = 0.70
    deltaT_min_gascooler: float = 5.0
    P_high_min: float = 8e6
    P_high_max: float = 50e6
    pressure_step: float = 1.0e6
    N_segments: int = 300


COMPONENTS = {
    1: "Saida lado frio IHX / entrada compressor",
    2: "Saida compressor / entrada gas cooler",
    3: "Saida gas cooler / entrada lado quente IHX",
    4: "Saida lado quente IHX / entrada valvula",
    5: "Saida valvula / entrada evaporador",
    6: "Saida evaporador / entrada lado frio IHX",
}


def water_side(inputs: CycleInputs) -> dict[str, float]:
    """Calcula propriedades do lado da agua e vazao de vapor."""
    P_steam = props("P", "T", inputs.T_steam, "Q", 0, FLUID_WATER)
    h_in = props("H", "T", inputs.T_feedwater, "P", P_steam, FLUID_WATER)
    h_out = props("H", "P", P_steam, "Q", 1, FLUID_WATER)
    m_steam = inputs.Q_sink / (h_out - h_in)
    return {
        "P_steam": P_steam,
        "h_water_in": h_in,
        "h_water_out": h_out,
        "m_steam": m_steam,
    }


def base_low_state(inputs: CycleInputs) -> dict[str, float]:
    P_low = props("P", "T", inputs.T_evap, "Q", 1, FLUID_CO2)
    h6 = props("H", "P", P_low, "Q", 1, FLUID_CO2)
    s6 = props("S", "P", P_low, "Q", 1, FLUID_CO2)
    return {"P_low": P_low, "state6": {"P": P_low, "T": inputs.T_evap, "h": h6, "s": s6, "x": 1.0}}


def evaluate_for_T3(P_high: float, T3: float, inputs: CycleInputs, water: dict[str, float]) -> dict:
    """Recalcula todo o ciclo para uma pressao alta e uma tentativa de T3."""
    low = base_low_state(inputs)
    P_low = low["P_low"]
    state6 = low["state6"]
    h6 = state6["h"]
    T6 = state6["T"]

    if P_high <= P_low:
        raise PropertyError("Pressao alta menor ou igual a pressao baixa.")
    if T3 <= T6:
        raise PropertyError("T3 menor ou igual a T6, impossibilitando troca no IHX.")

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

    dT_A = T3 - state1["T"]
    dT_B = state4["T"] - T6
    if dT_A <= 0 or dT_B <= 0:
        raise PropertyError("Cruzamento de temperatura no IHX.")

    h2s = props("H", "P", P_high, "S", state1["s"], FLUID_CO2)
    h2 = h1 + (h2s - h1) / inputs.eta_is_comp
    state2 = state_ph(P_high, h2, FLUID_CO2)

    if state2["T"] <= inputs.T_steam + inputs.deltaT_min_gascooler:
        raise PropertyError("Temperatura de descarga insuficiente para o approach minimo.")

    h_co2 = np.linspace(h3, h2, inputs.N_segments)
    h_water = np.linspace(water["h_water_in"], water["h_water_out"], inputs.N_segments)
    try:
        T_co2 = np.asarray(PropsSI("T", "P", P_high, "H", h_co2, FLUID_CO2), dtype=float)
        T_water = np.asarray(PropsSI("T", "P", water["P_steam"], "H", h_water, FLUID_WATER), dtype=float)
    except Exception as exc:
        raise PropertyError("Falha ao calcular perfil vetorizado do gas cooler.") from exc
    if not (np.all(np.isfinite(T_co2)) and np.all(np.isfinite(T_water))):
        raise PropertyError("Perfil do gas cooler contem temperaturas nao finitas.")
    deltaT = T_co2 - T_water
    min_idx = int(np.argmin(deltaT))
    min_approach = float(deltaT[min_idx])
    pinch_location = float(min_idx / (inputs.N_segments - 1))

    state5 = state_ph(P_low, h4, FLUID_CO2)
    x5 = quality_or_none(P_low, h4, FLUID_CO2)
    q_evap = h6 - h4
    q_gc = h2 - h3
    w_comp = h2 - h1

    if q_evap <= 0:
        raise PropertyError("Calor especifico no evaporador nao positivo.")
    if q_gc <= 0 or w_comp <= 0:
        raise PropertyError("Calor no gas cooler ou trabalho de compressao nao positivo.")

    m_co2 = inputs.Q_sink / q_gc
    W_shaft = m_co2 * w_comp
    W_electric = W_shaft / inputs.eta_motor
    COP = inputs.Q_sink / W_electric

    energy_balance_error = inputs.Q_sink - (m_co2 * q_evap + W_shaft)
    ihx_balance_error = (h3 - h4) - (h1 - h6)
    superheat = state1["T"] - props("T", "P", P_low, "Q", 1, FLUID_CO2)
    warnings = []
    if superheat < 5.0:
        warnings.append("Superaquecimento na entrada do compressor menor que 5 K.")

    return {
        "states": {1: state1, 2: state2, 3: state3, 4: state4, 5: state5, 6: state6},
        "P_high": P_high,
        "P_low": P_low,
        "P_steam": water["P_steam"],
        "q_ihx": q_ihx,
        "q_evap": q_evap,
        "q_gc": q_gc,
        "w_comp": w_comp,
        "m_co2": m_co2,
        "m_steam": water["m_steam"],
        "W_shaft": W_shaft,
        "W_electric": W_electric,
        "COP": COP,
        "superheat": superheat,
        "ihx_min_approach": min(dT_A, dT_B),
        "gas_cooler_min_approach": min_approach,
        "pinch_location": pinch_location,
        "energy_balance_error": energy_balance_error,
        "ihx_balance_error": ihx_balance_error,
        "warnings": warnings,
        "profiles": {
            "f": np.linspace(0.0, 1.0, inputs.N_segments),
            "T_co2_gc": T_co2,
            "T_water_gc": T_water,
            "deltaT_gc": deltaT,
        },
    }


def solve_pressure(P_high: float, inputs: CycleInputs) -> dict:
    """Resolve T3 pelo pinch do gas cooler para uma pressao alta."""
    water = water_side(inputs)
    Pcrit = PropsSI("PCRIT", FLUID_CO2)
    if P_high <= Pcrit:
        return invalid_result(P_high, f"Pressao alta abaixo da pressao critica do CO2 ({Pcrit / 1e6:.3f} MPa).")

    def residual(T3: float) -> float:
        return evaluate_for_T3(P_high, T3, inputs, water)["gas_cooler_min_approach"] - inputs.deltaT_min_gascooler

    T_low = max(inputs.T_evap + 0.5, inputs.T_feedwater + 0.5)
    T_high = min(inputs.T_steam + 90.0, 520.0)
    grid = np.linspace(T_low, T_high, 28)
    samples: list[tuple[float, float]] = []
    failures: list[str] = []
    for T in grid:
        try:
            samples.append((float(T), float(residual(float(T)))))
        except Exception as exc:
            failures.append(str(exc))

    bracket: Optional[tuple[float, float]] = None
    for (Ta, ra), (Tb, rb) in zip(samples, samples[1:]):
        if ra == 0:
            bracket = (Ta, Ta)
            break
        if ra * rb < 0:
            bracket = (Ta, Tb)
            break

    if bracket is None:
        reason = "Nao foi encontrado intervalo com mudanca de sinal para o pinch do gas cooler."
        if failures and not samples:
            reason += f" Primeira falha: {failures[0]}"
        return invalid_result(P_high, reason)

    try:
        if bracket[0] == bracket[1]:
            T3 = bracket[0]
        else:
            T3 = brentq(residual, bracket[0], bracket[1], xtol=1e-5, rtol=1e-8, maxiter=100)
        result = evaluate_for_T3(P_high, T3, inputs, water)
        result["valid"] = True
        result["invalid_reason"] = ""
        return result
    except Exception as exc:
        return invalid_result(P_high, str(exc))


def invalid_result(P_high: float, reason: str) -> dict:
    return {"P_high": P_high, "valid": False, "invalid_reason": reason}


def row_from_result(result: dict) -> dict:
    """Cria linha tabular para a varredura de pressoes."""
    if not result.get("valid", False):
        return {
            "P_high_MPa": mpa(result["P_high"]),
            "valid": False,
            "invalid_reason": result.get("invalid_reason", "invalido"),
        }

    s = result["states"]
    return {
        "P_high_MPa": mpa(result["P_high"]),
        "COP": result["COP"],
        "T1_C": celsius(s[1]["T"]),
        "T2_C": celsius(s[2]["T"]),
        "T3_C": celsius(s[3]["T"]),
        "T4_C": celsius(s[4]["T"]),
        "T5_C": celsius(s[5]["T"]),
        "T6_C": celsius(s[6]["T"]),
        "superheat_K": result["superheat"],
        "m_CO2_kg_s": result["m_co2"],
        "m_steam_kg_h": result["m_steam"] * 3600.0,
        "W_electric_kW": result["W_electric"] / 1e3,
        "Q_evaporator_kW": result["m_co2"] * result["q_evap"] / 1e3,
        "x5": s[5]["x"],
        "IHX_heat_kW": result["m_co2"] * result["q_ihx"] / 1e3,
        "IHX_min_approach_K": result["ihx_min_approach"],
        "gas_cooler_min_approach_K": result["gas_cooler_min_approach"],
        "pinch_location": result["pinch_location"],
        "valid": True,
        "invalid_reason": "; ".join(result["warnings"]),
    }


def state_table(result: dict) -> pd.DataFrame:
    rows = []
    for i in range(1, 7):
        st = result["states"][i]
        rows.append(
            {
                "Estado": i,
                "Componente/localizacao": COMPONENTS[i],
                "P_MPa": mpa(st["P"]),
                "T_C": celsius(st["T"]),
                "h_kJ_kg": kjkg(st["h"]),
                "s_kJ_kgK": kjkgk(st["s"]),
                "Qualidade": st["x"],
            }
        )
    return pd.DataFrame(rows)


def pressure_sweep(inputs: CycleInputs) -> tuple[pd.DataFrame, list[dict]]:
    pressures = np.arange(inputs.P_high_min, inputs.P_high_max + 0.5 * inputs.pressure_step, inputs.pressure_step)
    results = [solve_pressure(float(P), inputs) for P in pressures]
    table = pd.DataFrame([row_from_result(r) for r in results])
    return table, results


def refine_optimum(best_pressure: float, inputs: CycleInputs) -> dict:
    lower = max(best_pressure - inputs.pressure_step, PropsSI("PCRIT", FLUID_CO2) * 1.001, inputs.P_high_min)
    upper = min(best_pressure + inputs.pressure_step, inputs.P_high_max)

    def objective(P: float) -> float:
        r = solve_pressure(float(P), inputs)
        if not r.get("valid", False):
            return 1e6
        return -r["COP"]

    opt = minimize_scalar(objective, bounds=(lower, upper), method="bounded", options={"xatol": 2e3})
    refined = solve_pressure(float(opt.x), inputs)
    if refined.get("valid", False):
        return refined
    return solve_pressure(best_pressure, inputs)


def find_optimum(inputs: CycleInputs) -> tuple[pd.DataFrame, list[dict], dict]:
    table, results = pressure_sweep(inputs)
    valid = table[table["valid"] == True].copy()  # noqa: E712
    if valid.empty:
        raise RuntimeError("Nenhuma pressao valida encontrada na varredura.")
    best_pressure = float(valid.loc[valid["COP"].idxmax(), "P_high_MPa"]) * 1e6
    optimum = refine_optimum(best_pressure, inputs)
    return table, results, optimum


def sensitivity_analysis(epsilon_values: list[float], base_inputs: CycleInputs) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    rows = []
    curves: dict[float, pd.DataFrame] = {}
    for eps in epsilon_values:
        inputs = CycleInputs(**{**base_inputs.__dict__, "epsilon_IHX": eps})
        table, _, optimum = find_optimum(inputs)
        curves[eps] = table
        rows.append(
            {
                "epsilon_IHX": eps,
                "P_opt_MPa": mpa(optimum["P_high"]),
                "COP_opt": optimum["COP"],
                "T3_opt_C": celsius(optimum["states"][3]["T"]),
                "W_electric_kW": optimum["W_electric"] / 1e3,
            }
        )
    return pd.DataFrame(rows), curves
