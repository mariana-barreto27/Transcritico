"""Plota dados experimentais do artigo sobre curvas geradas pelo modelo principal.

Importante: este script nao altera `cycle_model.py`. Ele apenas troca, durante a
execucao da validacao, a funcao do lado da agua para representar agua liquida
aquecida, que e o caso experimental do artigo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cycle_model as cm
from property_functions import FLUID_WATER, props


OUTPUT_DIR = ROOT / "outputs" / "validation"
EXPERIMENTAL_CSV = Path(__file__).resolve().parent / "experimental_points_for_overlay.csv"


CASES = [
    {"case_id": "HW65p0_Tevapm6p4", "T_water_out_C": 65.0, "T_evap_C": -6.4, "figure": "Fig7"},
    {"case_id": "HW77p5_Tevapm6p4", "T_water_out_C": 77.5, "T_evap_C": -6.4, "figure": "Fig7"},
    {"case_id": "HW77p5_Tevap0p3", "T_water_out_C": 77.5, "T_evap_C": 0.3, "figure": "Fig7"},
    {"case_id": "HW90p0_Tevapm6p4", "T_water_out_C": 90.0, "T_evap_C": -6.4, "figure": "Fig6"},
    {"case_id": "HW90p0_Tevapm3p0", "T_water_out_C": 90.0, "T_evap_C": -3.0, "figure": "Fig6"},
    {"case_id": "HW90p0_Tevap0p3", "T_water_out_C": 90.0, "T_evap_C": 0.3, "figure": "Fig6"},
]


def liquid_hot_water_side(inputs: cm.CycleInputs) -> dict[str, float]:
    """Adapta o lado da agua para o experimento: agua liquida 20,5 C -> T_saida.

    O `cycle_model.py` espera nomes ligados a vapor (`P_steam`, `m_steam`).
    Aqui mantemos os mesmos nomes so para compatibilidade interna, mas eles
    representam agua liquida pressurizada na validacao.
    """
    T_water_out = inputs.T_steam
    P_sat_out = props("P", "T", T_water_out, "Q", 0, FLUID_WATER)
    P_water = max(3e5, P_sat_out + 1e5)
    h_in = props("H", "T", inputs.T_feedwater, "P", P_water, FLUID_WATER)
    h_out = props("H", "T", T_water_out, "P", P_water, FLUID_WATER)
    m_water = inputs.Q_sink / (h_out - h_in)
    return {
        "P_steam": P_water,
        "h_water_in": h_in,
        "h_water_out": h_out,
        "m_steam": m_water,
    }


def run_model_curves() -> pd.DataFrame:
    """Roda o modelo principal para as mesmas temperaturas operacionais do artigo."""
    original_water_side = cm.water_side
    cm.water_side = liquid_hot_water_side
    rows = []
    try:
        for case in CASES:
            inputs = cm.CycleInputs(
                Q_sink=120e3,
                T_steam=case["T_water_out_C"] + 273.15,
                T_feedwater=20.5 + 273.15,
                T_evap=case["T_evap_C"] + 273.15,
                eta_is_comp=0.70,
                eta_motor=0.95,
                epsilon_IHX=0.70,
                deltaT_min_gascooler=5.0,
                P_high_min=9e6,
                P_high_max=13e6,
                pressure_step=0.5e6,
                N_segments=300,
            )
            table, _ = cm.pressure_sweep(inputs)
            table["case_id"] = case["case_id"]
            table["figure"] = case["figure"]
            table["T_water_out_C"] = case["T_water_out_C"]
            table["T_evap_C"] = case["T_evap_C"]
            rows.append(table)
    finally:
        cm.water_side = original_water_side
    return pd.concat(rows, ignore_index=True)


def compare_points(model: pd.DataFrame, exp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = model[model["valid"] == True].copy()  # noqa: E712
    for _, point in exp.iterrows():
        candidates = valid[valid["case_id"] == point["case_id"]].copy()
        if candidates.empty:
            rows.append({**point.to_dict(), "status": "sem_curva_modelo"})
            continue
        candidates["distance"] = abs(candidates["P_high_MPa"] - point["P_high_MPa"])
        nearest = candidates.sort_values("distance").iloc[0]
        rows.append(
            {
                **point.to_dict(),
                "status": "comparado",
                "P_high_MPa_model_nearest": nearest["P_high_MPa"],
                "COP_model": nearest["COP"],
                "COP_error_percent": 100.0 * (nearest["COP"] - point["COP_exp"]) / point["COP_exp"],
            }
        )
    return pd.DataFrame(rows)


def plot_overlay(model: pd.DataFrame, exp: pd.DataFrame, figure: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    model_fig = model[(model["figure"] == figure) & (model["valid"] == True)].copy()  # noqa: E712
    exp_fig = exp[exp["figure"] == figure].copy()

    for case_id, group in model_fig.groupby("case_id"):
        group = group.sort_values("P_high_MPa")
        label = model_label(case_id, prefix="Modelo")
        ax.plot(group["P_high_MPa"], group["COP"], linewidth=1.8, label=label)

    markers = ["o", "s", "^", "D", "X", "P"]
    for idx, (case_id, group) in enumerate(exp_fig.groupby("case_id")):
        label = model_label(case_id, prefix="Experimental")
        ax.scatter(
            group["P_high_MPa"],
            group["COP_exp"],
            marker=markers[idx % len(markers)],
            s=58,
            edgecolor="black",
            linewidth=0.6,
            label=label,
            zorder=5,
        )

    ax.set_xlabel("Pressao alta / descarga [MPa]")
    ax.set_ylabel("COP de aquecimento [-]")
    ax.set_title("Modelo principal com pontos experimentais sobrepostos")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close(fig)


def model_label(case_id: str, prefix: str) -> str:
    for case in CASES:
        if case["case_id"] == case_id:
            return f"{prefix}: agua {case['T_water_out_C']:g} C, evap {case['T_evap_C']:g} C"
    return f"{prefix}: {case_id}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = run_model_curves()
    exp = pd.read_csv(EXPERIMENTAL_CSV)
    comparison = compare_points(model, exp)

    model.to_csv(OUTPUT_DIR / "modelo_principal_condicoes_experimentais.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "comparacao_modelo_vs_experimental.csv", index=False)
    plot_overlay(model, exp, "Fig6", "overlay_fig6_agua90C.png")
    plot_overlay(model, exp, "Fig7", "overlay_fig7_agua65_77p5C.png")

    print("\nGraficos de validacao por sobreposicao criados")
    print("-" * 72)
    print("O modelo principal foi usado; cycle_model.py nao foi alterado.")
    print(f"Pontos experimentais no CSV: {len(exp)}")
    print(f"Pontos do modelo calculados: {len(model)}")
    print("\nResumo dos erros COP nos pontos digitalizados:")
    cols = ["case_id", "P_high_MPa", "COP_exp", "COP_model", "COP_error_percent"]
    print(comparison[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nArquivos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
