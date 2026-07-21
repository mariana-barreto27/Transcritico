# Validacao por sobreposicao com dados do artigo

Esta pasta foi criada para validar o modelo sem alterar os arquivos principais do projeto. O script `run_validation.py` importa o `cycle_model.py`, varia apenas os dados de entrada e plota os pontos experimentais do artigo por cima das curvas calculadas pelo seu modelo.

O artigo usado como referencia e:

White, Yarrall, Cleland e Hedley (2002), *Modelling the performance of a transcritical CO2 heat pump for high temperature heating*.

## O que esta validacao faz

O modelo principal foi feito para produzir vapor saturado a 120 degC. O artigo, porem, apresenta dados experimentais para aquecimento de agua liquida ate:

- 65 degC
- 77,5 degC
- 90 degC

com evaporacao em:

- -6,4 degC
- -3,0 degC
- +0,3 degC

e pressao de descarga entre 90 e 130 bar.

Por isso, o script de validacao usa o seu `cycle_model.py` e, somente durante a execucao da validacao, adapta o lado da agua para representar agua liquida aquecida de 20,5 degC ate a temperatura experimental de saida. O arquivo principal do modelo nao e editado.

## Como rodar

Na pasta principal do projeto:

```bash
source .venv/bin/activate
python validation/run_validation.py
```

Os resultados sao salvos em:

```text
outputs/validation
```

## Arquivos gerados

- `modelo_principal_condicoes_experimentais.csv`: curvas calculadas pelo modelo principal nas temperaturas do artigo.
- `comparacao_modelo_vs_experimental.csv`: erro entre modelo e pontos experimentais digitalizados.
- `overlay_fig6_agua90C.png`: pontos experimentais da Fig. 6 sobrepostos ao modelo.
- `overlay_fig7_agua65_77p5C.png`: pontos experimentais da Fig. 7 sobrepostos ao modelo.

## Como incluir mais dados experimentais

Digite ou corrija pontos do artigo em:

```text
validation/experimental_points_for_overlay.csv
```

Use as colunas:

- `T_water_out_C`
- `T_evap_C`
- `P_high_MPa`
- `COP_exp`
- `COP_exp`

Os pontos ja preenchidos foram lidos aproximadamente das Figuras 6 e 7, porque o artigo nao fornece uma tabela numerica completa. Se voce digitalizar as figuras com uma ferramenta propria, substitua esses valores no CSV.

## Limitacoes

Esta ainda e uma validacao preliminar, porque o seu modelo atual:

- fixa a carga termica desejada;
- nao usa mapa real do compressor;
- nao calcula vazao de CO2 a partir do compressor real;
- nao inclui perdas de pressao;
- nao inclui perdas de calor do compressor;
- nao calcula UA real do gas cooler e do recuperador.

Por isso, a comparacao mais honesta nesta etapa e principalmente de tendencia e ordem de grandeza do COP. Para validar capacidade termica de forma rigorosa, seria necessario incluir as correlacoes experimentais do artigo para compressor, perda de calor, gas cooler e recuperador.
