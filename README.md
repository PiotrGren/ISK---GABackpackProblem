# Knapsack-GA (0/1)

Implementacja algorytmu genetycznego do problemu plecakowego (0/1) w Pythonie.  
Projekt nastawiony na **eksperymenty w dużej skali**, replikowalność i wygodne I/O (JSON/JSONL).

## Spis treści
1. [Struktura katalogów](#struktura-katalogów)
2. [Instalacja](#instalacja)
3. [Format danych](#format-danych)
4. [Uruchomienie](#uruchomienie-cli)
5. [Pliki konfiguracyjne](#pliki-konfiguracyjne-przykład)
6. [Wyniki](#wyniki)

## Struktura katalogów

```bash
projekt/
├───data/
│   └───instances/           # *.json / *.jsonl
├───src/
│   ├───io.py                # wczytywanie/zapisywanie JSON/JSONL
│   ├───model.py             # dataclasses: Item, Instance
│   ├───ga.py                # GA: selekcja, krzyżowanie, mutacja, naprawa/penalty
│   ├───fitness.py           # liczenie wartości, wagi, fitness
│   ├───runner.py            # pętla ewolucji + logowanie trace
│   └───cli.py               # run-ga --instance ... --config ...
├───experiments/
│   ├───configs/             # preset parametry do serii eksperymentów
│   ├───notebooks/           # analiza wyników (ipynb)
│   └───results/             # *.jsonl z uruchomień
├───tests/
│   ├───test_fitness.py
│   └───test_repair.py
├───README.md
└───requirements.txt
```


## Instalacja

Wymagany Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Opcjonalnie (do walidacji na małych instancjach):

```bash
pip install pulp ortools
```


## Format danych

Wybrany format danych to pliki **.json** oraz **.jsonl**.

Przykład pojedynczej instacji:

```json
{
  "capacity": 250.0,
  "items": [
    {"id": 1, "weight": 12.0, "value": 4.0},
    {"id": 2, "weight": 7.0, "value": 10.0}
  ],
  "meta": {"name": "toy-2", "source": "synthetic"}
}
```

Wiele instancji to plik formatu **.jsonl** gdzie każda linia ma strukturę jak powyżej.

## Uruchomienie (CLI)

> **UWAGA!:** Poniższe polecenia należy uruchomić z katalogu głównego projektu - czyli katalogu ***projekt/** według głównej [struktury katalogów](#struktura-katalogów).

Przykład (pojedynczy run):
```bash
python -m src.cli \
  --instance data/instances/toy-2.json \
  --config experiments/configs/base.json \
  --out experiments/results/base.jsonl
```

Nadpisanie parametrów z linii poleceń:
```bash
python -m src.cli \
  --instance data/instances/big-1000.json \
  --config experiments/configs/base.json \
  --pop 400 --pc 0.9 --pm 0.01 --elitism 2
```

Seria runów (wiele seedów) można skonfigurować w pliku config.

#### Lista dostępnych subargumentów

Aby sprawdzić listę dostępnych subarguemntów należy z katalogu głównego projektu wykonać polecenie:
```bash
python -m src.cli --help
```

Wynik komendy wygląda nastepująco:
```bash
Usage: python -m src.cli [OPTIONS]

 Główna komenda: przygotuj parametry, wczytaj instancje i odpal eksperymenty.

┌─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ --instance         -i                  PATH     Ścieżka do pliku *.json lub *.jsonl z instancjami (domyślnie:       │
│                                                 data\instances\big-05-inverse-correlation-n2200.json)               │
│                                                 [default: data\instances\big-05-inverse-correlation-n2200.json]     │
│ --config           -c                  PATH     Plik konfiguracyjny JSON (domyślnie: experiments\configs\base.json) │
│                                                 [default: experiments\configs\base.json]                            │
│ --out              -o                  PATH     Plik wynikowy *.jsonl (dopisywanie; domyślnie:                      │
│                                                 experiments\results\auto.jsonl)                                     │
│                                                 [default: experiments\results\auto.jsonl]                           │
│ --pop                                  INTEGER  population                                                          │
│ --pc                                   FLOAT    pc                                                                  │
│ --pm                                   FLOAT    pm (liczba); zamiast "1/n"                                          │
│ --elitism                              INTEGER  elitism                                                             │
│ --max-generations                      INTEGER  max_generations                                                     │
│ --runs                                 INTEGER  liczba uruchomień na instancję                                      │
│ --selection-type                       TEXT     selection.type: "tournament" lub "roulette"                         │
│ --selection-k                          INTEGER  selection.k (dla tournament)                                        │
│ --crossover                            TEXT     crossover: "one_point" lub "uniform"                                │
│ --mutation                             TEXT     mutation: "bit_flip"                                                │
│ --constraint-mode                      TEXT     constraint.mode: "repair" lub "penalty"                             │
│ --lambda                               FLOAT    constraint.lambda (dla penalty)                                     │
│ --early-patience                       INTEGER  early_stop.patience                                                 │
│ --early-delta                          FLOAT    early_stop.min_delta                                                │
│ --subset-mode                          TEXT     subset.mode: "none" | "random" | "first_k"                          │
│ --subset-size                          INTEGER  subset.size                                                         │
│ --subset-seed                          INTEGER  subset.seed                                                         │
│ --seeds-csv                            TEXT     Nadpisz seeds: np. "0,1,2,3"                                        │
│ --dry-run              --no-dry-run             Tylko wczytaj i zweryfikuj config/instancje - nie uruchamiaj GA     │
│                                                 [default: no-dry-run]                                               │
│ --time-limit                           FLOAT    Limit czasu w sekundach (0 = brak limitu) [default: 0.0]            │
│ --log-every                            INTEGER  Wypisuj postęp co N generacji (0 = brak) [default: 20]              │
│ --help                                          Show this message and exit.                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Pliki konfiguracyjne (przykład)

> experiments/configs/base.json
```json
{
  "population": 200,
  "pc": 0.9,
  "pm": "1/n",
  "elitism": 2,
  "selection": {"type": "tournament", "k": 3},
  "crossover": "one_point",
  "mutation": "bit_flip",
  "constraint": {"mode": "repair", "lambda": 10.0},
  "max_generations": 500,
  "early_stop": {"patience": 50, "min_delta": 0.0},
  "runs": 20,
  "seeds": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
}

```

## Wyniki

Każdy run jest dopisywany jako jedna linia JSON do pliku **experiments/results/*.jsonl**.

Zawiera m.in. najlepszą wartość, wagę, wykonalność, czas, numer generacji, oraz ślad przebiegu (best/avg per gen).
