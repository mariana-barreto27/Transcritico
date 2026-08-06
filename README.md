# Transcritico
Bomba de calor transcritica de CO2 com IHX

Projeto em Python para modelar uma bomba de calor transcritica de CO2 com trocador de calor interno, destinada a fornecer 120 kW para produzir vapor saturado a 120°C 

## Instalacao

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execucao

```bash
python3 main.py
```

Para rodar tambem a sensibilidade da efetividade do IHX:

```bash
python3 main.py --sensitivity
```

Os resultados sao salvos em `outputs/`, incluindo:

- `varredura_pressoes.csv`
- `estados_otimo.csv`
- `lado_agua.csv`
- graficos `*.png`
- `diagrama_ph_otimo.png`
- `diagrama_ps_otimo.png`
- `diagrama_ph_extremos_pressao.png`
- `diagrama_ps_extremos_pressao.png`
- `sensibilidade_ihx.csv`, quando `--sensitivity` e usado

## Hipoteses adotadas

- Perdas de pressao e perdas de calor para o ambiente sao desprezadas.
- O estado 6 e vapor saturado de CO2 na temperatura de evaporacao.
- A pressao baixa e calculada como pressao de saturacao do CO2 em `T_evap`.
- Por padrao, a pressao do vapor e calculada pelo CoolProp como saturacao da agua em 120 degC, ou seja, nao e pressao atmosferica.
- A compressao usa eficiencia isentropica constante.
- A potencia eletrica considera eficiencia do motor constante.
- O IHX e modelado por efetividade, limitada pelos dois lados termicos.
- O gas cooler/gerador de vapor e resolvido impondo pinch minimo de 5 K, verificado em 300 segmentos.

## Parametros

As entradas ficam na dataclass `CycleInputs`, em `cycle_model.py`:

```python
Q_sink = 120e3
T_steam = 120 + 273.15
T_feedwater = 20 + 273.15
T_evap = 0 + 273.15
eta_is_comp = 0.70
eta_motor = 0.95
epsilon_IHX = 0.70
deltaT_min_gascooler = 5.0
P_high_min = 8e6
P_high_max = 20e6
pressure_step = 0.25e6
N_segments = 300
```

`T_evap` tambem aceita valor em degC quando for escrito diretamente como numero pequeno. Por exemplo:

```python
T_evap = 25.0
```

sera interpretado como 25 degC. Para CO2 saturado, `T_evap` precisa ficar abaixo da temperatura critica do CO2, aproximadamente 31 degC. Entao `T_evap = 40.0` nao representa evaporacao saturada neste modelo.

Pela linha de comando, use:

```bash
python3 main.py --t-evap-c 25
```

Tambem e possivel testar vapor superaquecido. `--steam-sat-c` define a pressao de saturacao do vapor e `--steam-out-c` define a temperatura final de saida:

```bash
python3 main.py --steam-sat-c 120 --steam-out-c 150
```

Se `--steam-out-c` nao for informado, o modelo produz vapor saturado.

Para usar agua/vapor em pressao atmosferica:

```bash
python3 main.py --water-pressure-atm
```

Nesse caso, a saturacao da agua ocorre perto de 100 degC. Se a saida continuar em 120 degC, o resultado e vapor superaquecido a 1 atm, nao vapor saturado a 120 degC:

```bash
python3 main.py --water-pressure-atm --steam-out-c 120
```

Tambem da para escolher outra pressao absoluta da agua/vapor, em bar:

```bash
python3 main.py --water-pressure-bar 1.5
```

Tambem e possivel alterar a efetividade pela linha de comando:

```bash
python3 main.py --epsilon-ihx 0.80
```

Para varrer a pressao alta de 8 a 50 MPa:

```bash
python3 main.py --p-high-min 8 --p-high-max 50
```

Para aumentar a carga termica entregue, o que aumenta a vazao calculada de CO2:

```bash
python3 main.py --q-sink-kw 240
```

Para impor diretamente uma vazao de CO2 e deixar o codigo calcular o calor entregue:

```bash
python3 main.py --m-co2-kg-s 0.65
```

Para rodar com uma heat source no evaporador, informe a temperatura de entrada e saida da fonte. O codigo usa:

```text
T_evap = T_source_out - approach_evaporador
```

Exemplo: fonte entra a 25 degC, sai a 5 degC e approach minimo no evaporador de 5 K:

```bash
python3 main.py --source-in-c 25 --source-out-c 5 --source-approach-k 5
```

Nesse caso, o codigo usa `T_evap = 0 degC` e calcula a vazao necessaria da fonte de calor em `outputs/lado_fonte_calor.csv`.

## Interpretacao da pressao otima

O programa primeiro faz uma varredura de 8 a 20 MPa, com passo de 0,25 MPa. Para cada pressao alta, ele encontra `T3` de modo que o menor approach no gas cooler seja 5 K. Em seguida calcula IHX, compressor, valvula, evaporador, vazoes, potencia e COP.

A pressao otima e a pressao valida que maximiza o COP de aquecimento. Depois da varredura, a solucao e refinada perto da melhor pressao com `scipy.optimize.minimize_scalar`.

## Validacoes

Cada ponto e marcado como invalido quando ocorre algum destes problemas:

- estado termodinamico invalido no CoolProp;
- pressao alta abaixo da pressao critica do CO2;
- impossibilidade de resolver o pinch do gas cooler;
- temperatura de descarga insuficiente para aquecer o vapor com o approach minimo;
- cruzamento de temperatura no IHX;
- calor de evaporacao, calor no gas cooler ou trabalho de compressao nao positivo.

O resumo no terminal mostra os erros de balanco de energia do ciclo e do IHX.

## Limitacoes

O modelo e adequado para estudo preliminar do ciclo. Ele nao inclui perdas de pressao, eficiencia volumetrica, mapa real de compressor, limitacoes construtivas dos trocadores, queda de temperatura por fouling, controle operacional, custos ou analise exergetica.
