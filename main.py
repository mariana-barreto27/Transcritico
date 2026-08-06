from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cycle_model import CycleInputs, find_optimum, heat_source_side, sensitivity_analysis, state_table, water_side
from plots import (
    plot_gas_cooler_TQ,
    plot_ihx_profile,
    plot_ph_ps_extreme_pressures,
    plot_ph_ps_optimum,
    plot_pressure_curves,
    plot_sensitivity,
)
from property_functions import celsius, mpa


def print_summary(optimum: dict, states: pd.DataFrame, output_dir: Path, water: dict[str, float], source: dict[str, float]) -> None:
    print("\nResumo do caso otimo")
    print("-" * 72)
    print(f"Calor entregue:      {optimum['Q_sink'] / 1e3:.2f} kW")
    print(f"Pressao alta otima: {mpa(optimum['P_high']):.3f} MPa")
    print(f"Pressao baixa:       {mpa(optimum['P_low']):.3f} MPa")
    print(f"Pressao agua/vapor:  {mpa(water['P_steam']):.3f} MPa")
    print(f"T saturacao agua:    {celsius(water['T_steam_sat']):.2f} degC")
    print(f"T saida agua/vapor:  {celsius(water['T_steam_out']):.2f} degC")
    print(f"Superaq. vapor:      {water['steam_superheat']:.2f} K")
    print(f"COP de aquecimento:  {optimum['COP']:.3f}")
    print(f"Potencia eletrica:   {optimum['W_electric'] / 1e3:.2f} kW")
    print(f"Vazao de CO2:        {optimum['m_co2']:.4f} kg/s")
    print(f"Vazao agua/vapor:    {optimum['m_steam']:.4f} kg/s ({optimum['m_steam'] * 3600:.1f} kg/h)")
    print(f"T descarga comp.:    {celsius(optimum['states'][2]['T']):.2f} degC")
    print(f"T saida gas cooler:  {celsius(optimum['states'][3]['T']):.2f} degC")
    print(f"Superaquecimento:    {optimum['superheat']:.2f} K")
    print(f"Pinch gas cooler:    {optimum['gas_cooler_min_approach']:.3f} K em f={optimum['pinch_location']:.3f}")
    print(f"Min approach IHX:    {optimum['ihx_min_approach']:.3f} K")
    print(f"Erro balanco ciclo:  {optimum['energy_balance_error']:.6f} W")
    print(f"Erro balanco IHX:    {optimum['ihx_balance_error']:.6e} J/kg")
    if source:
        print("\nLado da fonte de calor")
        print(f"Fonte entra:         {source['T_source_in_C']:.2f} degC")
        print(f"Fonte sai:           {source['T_source_out_C']:.2f} degC")
        print(f"T evaporacao:        {source['T_evap_C']:.2f} degC")
        print(f"Approach evaporador: {source['evaporator_min_approach_K']:.2f} K")
        print(f"Calor da fonte:      {source['Q_source_kW']:.2f} kW")
        print(f"Vazao da fonte:      {source['m_source_kg_s']:.4f} kg/s ({source['m_source_kg_h']:.1f} kg/h)")
    if optimum["warnings"]:
        print("Avisos:              " + "; ".join(optimum["warnings"]))
    if water.get("water_mode") == "pressao_fixa_vazao_fixa_T_saida_calculada":
        print("Modo agua/vapor:     pressao fixa + vazao fixa; T saida calculada pelo balanco de energia")
    print("\nTabela de estados no ponto otimo")
    print(states.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nArquivos salvos em: {output_dir}")


def run(args: argparse.Namespace) -> None:
    if args.q_sink_kw is not None and args.m_co2_kg_s is not None:
        raise ValueError("Use --q-sink-kw OU --m-co2-kg-s. Os dois juntos deixam a escala do ciclo ambigua.")
    pressure_modes = [args.water_pressure_atm, args.water_pressure_bar is not None, args.water_pressure_mpa is not None]
    if sum(bool(mode) for mode in pressure_modes) > 1:
        raise ValueError("Use apenas uma pressao da agua: --water-pressure-atm, --water-pressure-bar ou --water-pressure-mpa.")
    if args.steam_out_c is not None and args.water_out_c is not None:
        raise ValueError("Use --water-out-c OU --steam-out-c, nao os dois juntos.")
    if args.m_water_kg_h is not None and args.m_water_kg_s is not None:
        raise ValueError("Use --m-water-kg-h OU --m-water-kg-s, nao os dois juntos.")
    if (args.m_water_kg_h is not None or args.m_water_kg_s is not None) and args.m_co2_kg_s is not None:
        raise ValueError("Use --m-water-kg-h/--m-water-kg-s OU --m-co2-kg-s. Os dois juntos deixam Q_sink ambiguo.")
    water_out_C = args.water_out_c if args.water_out_c is not None else args.steam_out_c
    if (args.m_water_kg_h is not None or args.m_water_kg_s is not None) and water_out_C is not None:
        raise ValueError("Com vazao de agua imposta, nao use temperatura de saida: ela sera calculada.")

    source_enabled = args.source_in_c is not None or args.source_out_c is not None
    if source_enabled and (args.source_in_c is None or args.source_out_c is None):
        raise ValueError("Para usar heat source, informe --source-in-c e --source-out-c.")
    if source_enabled and args.source_in_c <= args.source_out_c:
        raise ValueError("A heat source precisa entrar mais quente do que sai: --source-in-c > --source-out-c.")

    T_evap_C = args.t_evap_c
    if source_enabled:
        T_evap_C = args.source_out_c - args.source_approach_k
    P_steam = None
    if args.water_pressure_atm:
        P_steam = 101325.0
    elif args.water_pressure_bar is not None:
        P_steam = args.water_pressure_bar * 1e5
    elif args.water_pressure_mpa is not None:
        P_steam = args.water_pressure_mpa * 1e6
    m_water_target = None
    if args.m_water_kg_h is not None:
        if args.m_water_kg_h <= 0:
            raise ValueError("--m-water-kg-h precisa ser positivo.")
        m_water_target = args.m_water_kg_h / 3600.0
    elif args.m_water_kg_s is not None:
        if args.m_water_kg_s <= 0:
            raise ValueError("--m-water-kg-s precisa ser positivo.")
        m_water_target = args.m_water_kg_s

    inputs = CycleInputs(
        Q_sink=args.q_sink_kw * 1e3 if args.q_sink_kw is not None else 120e3,
        m_co2_target=args.m_co2_kg_s,
        m_water_target=m_water_target,
        P_steam=P_steam,
        epsilon_IHX=args.epsilon_ihx,
        T_steam=args.steam_sat_c + 273.15,
        T_steam_out=water_out_C + 273.15 if water_out_C is not None else None,
        T_feedwater=args.feedwater_c + 273.15,
        T_evap=T_evap_C + 273.15,
        T_source_in=args.source_in_c + 273.15 if source_enabled else None,
        T_source_out=args.source_out_c + 273.15 if source_enabled else None,
        cp_source=args.source_cp_kj_kgk * 1e3,
        deltaT_min_evaporator=args.source_approach_k,
        P_high_min=args.p_high_min * 1e6,
        P_high_max=args.p_high_max * 1e6,
        pressure_step=args.pressure_step_mpa * 1e6,
        N_segments=args.segments,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "\nRodando varredura: "
        f"P alta {args.p_high_min:.2f} a {args.p_high_max:.2f} MPa, "
        f"passo {args.pressure_step_mpa:.2f} MPa, "
        f"{args.segments} segmentos no gas cooler.",
        flush=True,
    )
    if args.p_high_max > 20.0 and args.pressure_step_mpa <= 0.25 and args.segments >= 300:
        print(
            "Esse caso e pesado e pode levar alguns minutos. "
            "Para testar rapido, use --pressure-step-mpa 1 --segments 100.",
            flush=True,
        )

    sweep_table, sweep_results, optimum = find_optimum(inputs)
    states = state_table(optimum)
    water = {
        **water_side(inputs),
        "Q_sink": optimum["Q_sink"],
        "m_steam": optimum["m_steam"],
    }
    source = heat_source_side(optimum, inputs)

    sweep_table.to_csv(output_dir / "varredura_pressoes.csv", index=False)
    states.to_csv(output_dir / "estados_otimo.csv", index=False)
    pd.DataFrame([water]).to_csv(output_dir / "lado_agua.csv", index=False)
    if source:
        pd.DataFrame([source]).to_csv(output_dir / "lado_fonte_calor.csv", index=False)

    plot_pressure_curves(sweep_table, output_dir)
    plot_gas_cooler_TQ(optimum, output_dir)
    plot_ihx_profile(optimum, output_dir)
    plot_ph_ps_extreme_pressures(sweep_results, output_dir)
    plot_ph_ps_optimum(optimum, output_dir)

    if args.sensitivity:
        eps_values = [0.50, 0.60, 0.70, 0.80, 0.90]
        sens_table, curves = sensitivity_analysis(eps_values, inputs)
        sens_table.to_csv(output_dir / "sensibilidade_ihx.csv", index=False)
        for eps, curve in curves.items():
            curve.to_csv(output_dir / f"varredura_epsilon_{eps:.2f}.csv", index=False)
        plot_sensitivity(curves, output_dir)

    print_summary(optimum, states, output_dir, water, source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modelo de bomba de calor transcritica de CO2 com IHX.")
    parser.add_argument("--epsilon-ihx", type=float, default=0.70, help="Efetividade do IHX.")
    parser.add_argument("--q-sink-kw", type=float, default=None, help="Calor entregue no gas cooler/gerador de vapor [kW].")
    parser.add_argument("--m-co2-kg-s", type=float, default=None, help="Forca uma vazao de CO2 [kg/s] e calcula o calor/vapor resultante.")
    parser.add_argument(
        "--m-water-kg-h",
        type=float,
        default=None,
        help="Forca uma vazao de agua/vapor [kg/h] e calcula a temperatura final pelo balanco de energia.",
    )
    parser.add_argument(
        "--m-water-kg-s",
        type=float,
        default=None,
        help="Forca uma vazao de agua/vapor [kg/s] e calcula a temperatura final pelo balanco de energia.",
    )
    parser.add_argument(
        "--steam-sat-c",
        type=float,
        default=120.0,
        help="Temperatura de saturacao usada quando a pressao da agua/vapor nao e informada [degC].",
    )
    parser.add_argument("--steam-out-c", type=float, default=None, help="Nome antigo para --water-out-c.")
    parser.add_argument("--water-out-c", type=float, default=None, help="Temperatura de saida da agua/vapor [degC].")
    parser.add_argument("--water-pressure-atm", action="store_true", help="Usa agua/vapor a 1 atm absoluto.")
    parser.add_argument("--water-pressure-bar", type=float, default=None, help="Pressao absoluta da agua/vapor [bar].")
    parser.add_argument("--water-pressure-mpa", type=float, default=None, help="Pressao absoluta da agua/vapor [MPa].")
    parser.add_argument("--feedwater-c", type=float, default=20.0, help="Temperatura da agua de alimentacao [degC].")
    parser.add_argument("--t-evap-c", type=float, default=0.0, help="Temperatura de evaporacao quando nao ha heat source [degC].")
    parser.add_argument("--source-in-c", type=float, default=None, help="Temperatura de entrada da heat source no evaporador [degC].")
    parser.add_argument("--source-out-c", type=float, default=None, help="Temperatura de saida da heat source no evaporador [degC].")
    parser.add_argument("--source-approach-k", type=float, default=5.0, help="Approach minimo no evaporador entre fonte e CO2 [K].")
    parser.add_argument("--source-cp-kj-kgk", type=float, default=4.18, help="Calor especifico medio da heat source [kJ/(kg K)].")
    parser.add_argument("--p-high-min", type=float, default=8.0, help="Pressao alta minima da varredura [MPa].")
    parser.add_argument("--p-high-max", type=float, default=50.0, help="Pressao alta maxima da varredura [MPa].")
    parser.add_argument("--pressure-step-mpa", type=float, default=0.25, help="Passo da varredura de pressao alta [MPa].")
    parser.add_argument("--segments", type=int, default=300, help="Numero de segmentos no perfil T-Q do gas cooler.")
    parser.add_argument("--sensitivity", action="store_true", help="Executa sensibilidade da efetividade do IHX.")
    parser.add_argument("--output-dir", default="outputs", help="Pasta para tabelas e graficos.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
