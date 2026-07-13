"""Funcoes auxiliares para propriedades termodinamicas com CoolProp."""

from __future__ import annotations

import math
from typing import Optional

from CoolProp.CoolProp import PropsSI


FLUID_CO2 = "CO2"
FLUID_WATER = "Water"


class PropertyError(ValueError):
    """Erro usado quando o CoolProp nao consegue calcular um estado fisico."""


def props(output: str, name1: str, value1: float, name2: str, value2: float, fluid: str) -> float:
    """Calcula uma propriedade e converte excecoes do CoolProp para mensagens claras."""
    try:
        result = PropsSI(output, name1, value1, name2, value2, fluid)
    except Exception as exc:  # CoolProp lanca tipos diferentes conforme o backend.
        raise PropertyError(
            f"Estado invalido para {fluid}: {output}({name1}={value1:.6g}, {name2}={value2:.6g})"
        ) from exc
    if not math.isfinite(result):
        raise PropertyError(
            f"CoolProp retornou valor nao finito para {fluid}: "
            f"{output}({name1}={value1:.6g}, {name2}={value2:.6g})"
        )
    return float(result)


def state_ph(P: float, h: float, fluid: str = FLUID_CO2) -> dict[str, Optional[float]]:
    """Retorna estado a partir de pressao e entalpia."""
    T = props("T", "P", P, "H", h, fluid)
    s = props("S", "P", P, "H", h, fluid)
    q = quality_or_none(P, h, fluid)
    return {"P": P, "T": T, "h": h, "s": s, "x": q}


def state_pt(P: float, T: float, fluid: str = FLUID_CO2) -> dict[str, Optional[float]]:
    """Retorna estado a partir de pressao e temperatura."""
    h = props("H", "P", P, "T", T, fluid)
    s = props("S", "P", P, "T", T, fluid)
    q = quality_or_none(P, h, fluid)
    return {"P": P, "T": T, "h": h, "s": s, "x": q}


def quality_or_none(P: float, h: float, fluid: str = FLUID_CO2) -> Optional[float]:
    """Retorna qualidade quando o estado esta em duas fases; caso contrario, None."""
    try:
        q = PropsSI("Q", "P", P, "H", h, fluid)
    except Exception:
        return None
    if not math.isfinite(q) or q < -1e-7 or q > 1.0 + 1e-7:
        return None
    return min(1.0, max(0.0, float(q)))


def celsius(T: float) -> float:
    return T - 273.15


def mpa(P: float) -> float:
    return P / 1e6


def kjkg(h: float) -> float:
    return h / 1e3


def kjkgk(s: float) -> float:
    return s / 1e3
