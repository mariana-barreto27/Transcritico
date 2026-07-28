from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cycle_model import CycleInputs, find_optimum, sensitivity_analysis, state_table, water_side
from plots import plot_gas_cooler_TQ, plot_ihx_profile, plot_ph_ps_extreme_pressures, plot_pressure_curves, plot_sensitivity
from property_functions import celsius, mpa


def print_summary(optimum: dict, states: pd.DataFrame, output_dir: Path) -> None:
    water = {
        "P_steam": optimum["P_steam"],
        "m_steam": optimum["m_steam"],
    }
    print("\nResumo do caso otimo")
    print("-" * 72)
    print(f"Pressao alta otima: {mpa(optimum['P_high']):.3f} MPa")
    print(f"Pressao baixa:       {mpa(optimum['P_low']):.3f} MPa")
    print(f"Pressao do vapor:    {mpa(water['P_steam']):.3f} MPa")
    print(f"COP de aquecimento:  {optimum['COP']:.3f}")
    print(f"Potencia eletrica:   {optimum['W_electric'] / 1e3:.2f} kW")
    print(f"Vazao de CO2:        {optimum['m_co2']:.4f} kg/s")
    print(f"Vazao de vapor:      {water['m_steam']:.4f} kg/s ({water['m_steam'] * 3600:.1f} kg/h)")
    print(f"T descarga comp.:    {celsius(optimum['states'][2]['T']):.2f} degC")
    print(f"T saida gas cooler:  {celsius(optimum['states'][3]['T']):.2f} degC")
    print(f"Superaquecimento:    {optimum['superheat']:.2f} K")
    print(f"Pinch gas cooler:    {optimum['gas_cooler_min_approach']:.3f} K em f={optimum['pinch_location']:.3f}")
    print(f"Min approach IHX:    {optimum['ihx_min_approach']:.3f} K")
    print(f"Erro balanco ciclo:  {optimum['energy_balance_error']:.6f} W")
    print(f"Erro balanco IHX:    {optimum['ihx_balance_error']:.6e} J/kg")
    if optimum["warnings"]:
        print("Avisos:              " + "; ".join(optimum["warnings"]))
    print("\nTabela de estados no ponto otimo")
    print(states.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nArquivos salvos em: {output_dir}")


def run(args: argparse.Namespace) -> None:
    inputs = CycleInputs(epsilon_IHX=args.epsilon_ihx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_table, sweep_results, optimum = find_optimum(inputs)
    states = state_table(optimum)
    water = water_side(inputs)

    sweep_table.to_csv(output_dir / "varredura_pressoes.csv", index=False)
    states.to_csv(output_dir / "estados_otimo.csv", index=False)
    pd.DataFrame([water]).to_csv(output_dir / "lado_agua.csv", index=False)

    plot_pressure_curves(sweep_table, output_dir)
    plot_gas_cooler_TQ(optimum, output_dir)
    plot_ihx_profile(optimum, output_dir)
    plot_ph_ps_extreme_pressures(sweep_results, output_dir)

    if args.sensitivity:
        eps_values = [0.50, 0.60, 0.70, 0.80, 0.90]
        sens_table, curves = sensitivity_analysis(eps_values, inputs)
        sens_table.to_csv(output_dir / "sensibilidade_ihx.csv", index=False)
        for eps, curve in curves.items():
            curve.to_csv(output_dir / f"varredura_epsilon_{eps:.2f}.csv", index=False)
        plot_sensitivity(curves, output_dir)

    print_summary(optimum, states, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modelo de bomba de calor transcritica de CO2 com IHX.")
    parser.add_argument("--epsilon-ihx", type=float, default=0.70, help="Efetividade do IHX.")
    parser.add_argument("--sensitivity", action="store_true", help="Executa sensibilidade da efetividade do IHX.")
    parser.add_argument("--output-dir", default="outputs", help="Pasta para tabelas e graficos.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
