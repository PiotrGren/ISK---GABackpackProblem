"""
Plik: src/cli.py

Cel i rola w projekcie
----------------------
Interfejs wiersza poleceń (CLI) do uruchamiania eksperymentów:
- Wczytuje plik konfiguracyjny (np. `experiments/configs/base.json`),
- Pozwala *nadpisać* wybrane parametry z linii poleceń (np. `--pop`, `--pc` itd.),
- Wczytuje instancje z `data/instances/*.json` lub `*.jsonl` (strumieniowo),
- Stosuje opcjonalny *subset* (random/first_k) zgodnie z configiem/flagami,
- W trybie produkcyjnym wywoła pętlę ewolucji z `runner.py` i dopisze wyniki do
  `experiments/results/*.jsonl`.

Jak łączy się z resztą:
- Używa `src/io.py` do I/O i `src/model.py` do walidacji configu,
- Do faktycznego uruchomienia algorytmu zaimportuje `runner.run_experiment`.
  Dopóki `runner.py` nie jest gotowy, CLI zadziała w trybie walidacji/dry-run
  i poda jasny komunikat, co jeszcze trzeba dodać.

Powiązanie z projektem:
- Komenda przewodnia: `python -m src.cli run-ga --instance ... --config ... --out ...`
- Dzięki nadpisaniom można szybko robić siatki parametrów bez pisania nowych plików.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import typer

# Opcjonalnie: ładniejsze printy, ale nie jest wymagane do działania
try:
    from rich import print      # type: ignore
except Exception:               # pragma: no cover
    pass

from .model import Params, SubsetConfig
from .io import read_json, iter_instances, apply_subset


app = typer.Typer(add_completion=False, help="CLI do uruchamiania GA dla problemu plecakowego.")


def _merge_overrides(
    params: Params,
    pop: Optional[int],
    pc: Optional[float],
    pm: Optional[float],
    elitism: Optional[int],
    max_generations: Optional[int],
    runs: Optional[int],
    subset_mode: Optional[str],
    subset_size: Optional[int],
    subset_seed: Optional[int],
    selection_type: Optional[str],
    selection_k: Optional[int],
    crossover: Optional[str],
    mutation: Optional[str],
    constraint_mode: Optional[str],
    lambda_: Optional[float],
    early_patience: Optional[int],
    early_delta: Optional[float],
    seeds_csv: Optional[str]
    ) -> Params:
    """Zastosuj ewentualne nadpisania z linii poleceń do obiektu Params."""
    data = params.model_dump(by_alias=True)

    if pop is not None:
        data["population"] = pop
    if pc is not None:
        data["pc"] = pc
    if pm is not None:
        data["pm"] = float(pm)
    if elitism is not None:
        data["elitism"] = elitism
    if max_generations is not None:
        data["max_generations"] = max_generations
    if runs is not None:
        data["runs"] = runs

    if selection_type is not None:
        data["selection"]["type"] = selection_type
    if selection_k is not None:
        data["selection"]["k"] = selection_k

    if crossover is not None:
        data["crossover"] = crossover
    if mutation is not None:
        data["mutation"] = mutation

    if constraint_mode is not None:
        data["constraint"]["mode"] = constraint_mode
    if lambda_ is not None:
        data["constraint"]["lambda"] = lambda_

    if early_patience is not None:
        data["early_stop"]["patience"] = early_patience
    if early_delta is not None:
        data["early_stop"]["min_delta"] = early_delta

    if subset_mode is not None:
        data["subset"]["mode"] = subset_mode
    if subset_size is not None:
        data["subset"]["size"] = subset_size
    if subset_seed is not None:
        data["subset"]["seed"] = subset_seed

    if seeds_csv:
        seeds = [int(s) for s in seeds_csv.split(",") if s.strip()]
        if seeds:
            data["seeds"] = seeds

    return Params.model_validate(data)


DEFAULT_INSTANCE = Path("data/instances/big-05-inverse-correlation-n2200.json")
DEFAULT_CONFIG = Path("experiments/configs/base.json")
DEFAULT_OUT = Path("experiments/results/auto.jsonl")


@app.command("run-ga")
def run_ga(
    instance: Path = typer.Option(
        DEFAULT_INSTANCE, "--instance", "-i",
        help=f"Ścieżka do pliku *.json lub *.jsonl z instancjami (domyślnie: {DEFAULT_INSTANCE})"
    ),
    config: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c",
        help=f"Plik konfiguracyjny JSON (domyślnie: {DEFAULT_CONFIG})"
    ),
    out: Path = typer.Option(
        DEFAULT_OUT, "--out", "-o",
        help=f"Plik wynikowy *.jsonl (dopisywanie; domyślnie: {DEFAULT_OUT})"
    ),

    # Nadpisania popularnych parametrów:
    pop: Optional[int] = typer.Option(None, help="population"),
    pc: Optional[float] = typer.Option(None, help="pc"),
    pm: Optional[float] = typer.Option(None, help='pm (liczba); zamiast "1/n"'),
    elitism: Optional[int] = typer.Option(None, help="elitism"),
    max_generations: Optional[int] = typer.Option(None, help="max_generations"),
    runs: Optional[int] = typer.Option(None, help="liczba uruchomień na instancję"),

    # Selekcja / operatory
    selection_type: Optional[str] = typer.Option(None, help='selection.type: "tournament" lub "roulette"'),
    selection_k: Optional[int] = typer.Option(None, help="selection.k (dla tournament)"),
    crossover: Optional[str] = typer.Option(None, help='crossover: "one_point" lub "uniform"'),
    mutation: Optional[str] = typer.Option(None, help='mutation: "bit_flip"'),

    # Constraint
    constraint_mode: Optional[str] = typer.Option(None, help='constraint.mode: "repair" lub "penalty"'),
    lambda_: Optional[float] = typer.Option(None, "--lambda", help="constraint.lambda (dla penalty)"),

    # Early-stop
    early_patience: Optional[int] = typer.Option(None, help="early_stop.patience"),
    early_delta: Optional[float] = typer.Option(None, help="early_stop.min_delta"),

    # Subset
    subset_mode: Optional[str] = typer.Option(None, help='subset.mode: "none" | "random" | "first_k"'),
    subset_size: Optional[int] = typer.Option(None, help="subset.size"),
    subset_seed: Optional[int] = typer.Option(None, help="subset.seed"),

    # Seeds lista
    seeds_csv: Optional[str] = typer.Option(None, help='Nadpisz seeds: np. "0,1,2,3"'),

    # Walidacja bez uruchamiania
    dry_run: bool = typer.Option(False, help="Tylko wczytaj i zweryfikuj config/instancje - nie uruchamiaj GA"),
):
    """Główna komenda: przygotuj parametry, wczytaj instancje i odpal eksperymenty."""
    # 1) Wczytaj config i zwaliduj
    cfg_dict = read_json(config)
    params = Params.model_validate(cfg_dict)

    # 2) Zastosuj ewentualne nadpisania z CLI
    params = _merge_overrides(
        params, pop, pc, pm, elitism, max_generations, runs,
        subset_mode, subset_size, subset_seed, selection_type, selection_k,
        crossover, mutation, constraint_mode, lambda_,
        early_patience, early_delta, seeds_csv
    )

    print("[bold]Konfiguracja końcowa (parsowana i zwalidowana):[/bold]")       # type: ignore
    print(params.model_dump(mode="json", by_alias=True))                        # type: ignore

    # 3) Oblicz ile instancji wczytamy (bez ładowania wszystkich do pamięci na raz)
    count = 0
    first_meta = None
    for inst in iter_instances(instance):
        inst2 = apply_subset(inst, params.subset)
        count += 1
        if first_meta is None:
            first_meta = inst2.meta
    print(f"[green]Znaleziono instancji:[/green] {count}")                      # type: ignore
    if first_meta:
        print(f"Przykładowa meta pierwszej instancji: {first_meta}")            # type: ignore

    if dry_run:
        print("[yellow]Dry-run zakończony. Nie uruchamiam GA (runner.py jeszcze nie jest podłączony).[/yellow]")        # type: ignore
        raise typer.Exit(code=0)

    # 4) Uruchomienie właściwego eksperymentu (jeśli runner.py jest gotowy)
    try:
        from .runner import run_experiment
    except Exception as e:  # pragma: no cover
        print("[red]Brak implementacji runner.run_experiment() lub błąd importu.[/red]")            # type: ignore
        print("Najpierw zaimplementuj `src/runner.py` (pętla ewolucji i logowanie wyników).")       # type: ignore
        print(f"Szczegóły importu: {e}")                                                            # type: ignore
        raise typer.Exit(code=1)

    # 5) Wywołanie: runner sam strumieniuje instancje i dopisuje do pliku wynikowego
    run_experiment(instance_path=instance, params=params, out_path=out)
    print(f"[bold green]Zakończono. Wyniki w:[/bold green] {out}")                                  # type: ignore
    
    
if __name__ == "__main__":
    app()