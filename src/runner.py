"""
Plik: src/runner.py

Cel i rola w projekcie
----------------------
To jest „kierownik” uruchomień (high-level runner) dla całego projektu.
Ten moduł:
1) Wczytuje instancje z plików *.json lub *.jsonl (strumieniowo) poprzez `io.iter_instances`.
2) Stosuje opcjonalne subsetowanie przedmiotów (np. test na 1500/5000 itemów) przez `io.apply_subset`.
3) Dla każdej instancji wykonuje serię uruchomień (`Params.runs`) z różnymi seedami (`Params.seeds`).
4) Dla każdego uruchomienia:
   - inicjalizuje generator losowy NumPy (deterministycznie z seed),
   - uruchamia pętlę ewolucji do `max_generations`,
   - w każdej generacji liczy fitness (przez `fitness.evaluate_population`) i tworzy nowe pokolenie (przez `ga.next_generation`),
   - realizuje early-stop (patience/min_delta),
   - zbiera trace (best/avg per generacja) jeżeli włączone w `Params.trace`.
5) Zapisuje wynik każdego uruchomienia do JSONL poprzez `io.write_run_result`
   (1 linia = 1 run), co ułatwia późniejszą analizę w rozdziale 4 sprawozdania.

Jak łączy się z resztą:
- `cli.py` woła tylko `run_experiment(...)`.
- `io.py` obsługuje wczytywanie instancji oraz zapis wyników JSONL.
- `fitness.py` liczy fitness i realizuje naprawę/karę.
- `ga.py` tworzy nowe pokolenie.

Założenia:
- Populacja to macierz (P,n) dtype=np.int8 z wartościami 0/1 (nie bool).
- Celem jest maksymalizacja wartości (value) przy ograniczeniu wagi (capacity).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
try:
    from rich.console import Console  # type: ignore
    console = Console()
except Exception:  # pragma: no cover
    console = None

from .model import Params, Instance
from .io import iter_instances, apply_subset, write_run_result
from .fitness import build_item_arrays, evaluate_population, population_values
from .ga import init_population, next_generation


# --- Pomocnicze: kodowanie chromosomu do JSON ----------------------------------------------------------------------
def bits_to_str(bits: np.ndarray) -> str:
    """Zamień wektor 0/1 na krótki zapis tekstowy '010101...' (mniejsze wyniki w JSONL)."""
    # bits jest np.int8, więc mapujemy na znaki
    return "".join("1" if b else "0" for b in bits.tolist())



# --- Pomocnicze: wybór najlepszego osobnika ------------------------------------------------------------------------
def best_of_population(pop: np.ndarray, fitness: np.ndarray) -> Tuple[int, float]:
    """Zwróć (index_best, best_fitness)."""
    i = int(np.argmax(fitness))
    return i, float(fitness[i])



# --- Pojedynczy run GA dla jednej instancji ------------------------------------------------------------------------
def run_single_ga(instance: Instance, params: Params, seed: int, time_limit_sec: float = 0.0, log_every: int = 50, run_index: int = 0) -> Dict[str, Any]:
    """
    Uruchom GA dla pojedynczej instancji i pojedynczego seeda.

    Zwraca słownik gotowy do zapisania jako 1 linia w JSONL.
    """
    stopped_reason = "max_generations"
    t0 = time.time()

    # 1) Przygotowanie danych instancji (NumPy arrays)
    weights, values = build_item_arrays(instance)
    capacity = float(instance.capacity)
    n = int(weights.shape[0])

    # 2) RNG deterministyczny
    rng = np.random.default_rng(seed)

    # 3) Populacja startowa
    pop = init_population(params.population, n, rng)  # (P,n) int8

    # 4) Evaluate początkowe
    pop, fitness, w_sum, v_sum = evaluate_population(
        pop=pop,
        weights=weights,
        values=values,
        capacity=capacity,
        constraint_mode=params.constraint.mode,
        lambda_=params.constraint.lambda_,
    )

    # 5) Trace (opcjonalnie)
    trace_best: List[float] = []
    trace_avg: List[float] = []

    # 6) Tracking best global
    best_idx, best_fit = best_of_population(pop, fitness)
    best_bits = pop[best_idx].copy()
    best_weight = float(w_sum[best_idx])
    best_value = float(v_sum[best_idx])

    # 7) Early-stop bookkeeping
    patience = int(params.early_stop.patience)
    min_delta = float(params.early_stop.min_delta)
    no_improve = 0
    best_ref = best_fit  # referencja do porównania poprawy

    # 8) Pętla generacji
    gen_reached = 0
    for gen in range(params.max_generations):
        gen_reached = gen + 1
        
        if log_every > 0 and (gen == 0 or gen_reached % log_every == 0):
          msg = (
            f"[[bold yellow]Generation[/bold yellow]] [bold white]{gen_reached}/{params.max_generations}[/bold white]  "
            f"[bold green]best_fit[/bold green] = [white]{best_fit:.3f}[/white]  "
            f"[bold green]best_w[/bold green] = [white]{best_weight:.2f}/{capacity:.2f}[/white] [cyan](≈{best_weight/capacity*100:.3f}%)[/cyan]  "
            f"[bold green]elapsed[/bold green] = [white]{time.time() - t0:.1f}s[/white]"
          )
          if console:
            console.print(msg)
          else:
            print(msg)

        # log trace (na podstawie obecnej populacji)
        if params.trace.store_best_per_gen:
            trace_best.append(float(np.max(fitness)))
        if params.trace.store_avg_per_gen:
            trace_avg.append(float(np.mean(fitness)))

        # aktualizacja global best
        cur_best_idx, cur_best_fit = best_of_population(pop, fitness)
        if cur_best_fit > best_fit:
            best_fit = cur_best_fit
            best_bits = pop[cur_best_idx].copy()
            best_weight = float(w_sum[cur_best_idx])
            best_value = float(v_sum[cur_best_idx])

        # early-stop: jeśli brak poprawy o min_delta przez patience generacji
        if patience > 0:
            if cur_best_fit > best_ref + min_delta:
                best_ref = cur_best_fit
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                  stopped_reason = "early_stop"
                  if console:
                    console.print(f"[[red]STOPPED[/red]][white]: Early stopping constraint reached.[/white]")
                  else:
                    print("[STOPPED]: Early stopping constraint reached.")
                  break

        if time_limit_sec and (time.time() - t0) >= time_limit_sec:
          stopped_reason = "time_limit"
          if console:
            console.print(f"[[red]STOPPED[/red]][white]: Time limit reached.[/white]")
          break

        # next generation
        pop_next = next_generation(pop, fitness, params, rng)

        # evaluate nowej populacji
        pop, fitness, w_sum, v_sum = evaluate_population(
            pop=pop_next,
            weights=weights,
            values=values,
            capacity=capacity,
            constraint_mode=params.constraint.mode,
            lambda_=params.constraint.lambda_,
        )

    # 9) Finalny feasibility (waga <= capacity)
    feasible = bool(best_weight <= capacity + 1e-9)

    t1 = time.time()
    elapsed = float(t1 - t0)

    # 10) Budowa wyniku do JSONL
    result: Dict[str, Any] = {
        "instance_meta": instance.meta or {},
        "capacity": capacity,
        "n_items": n,

        "seed": seed,
        "params": params.model_dump(mode="json", by_alias=True),

        "gen_reached": gen_reached,
        "time_sec": elapsed,

        "best_fitness": float(best_fit),
        "best_value": float(best_value),
        "best_weight": float(best_weight),
        "feasible": feasible,

        "best_bits": bits_to_str(best_bits),
        
        "time_limit_sec": float(time_limit_sec),
        "stopped_reason": stopped_reason,
    }

    # Trace dopisujemy tylko jeśli włączony (żeby wyniki nie były gigantyczne)
    if params.trace.store_best_per_gen:
        result["trace_best_fitness"] = trace_best
    if params.trace.store_avg_per_gen:
        result["trace_avg_fitness"] = trace_avg

    return result


# --- Publiczny interfejs: uruchom eksperymenty ---------------------------------------------------------------------
def run_experiment(instance_path: Path, params: Params, out_path: Path, time_limit_sec: float = 0.0, log_every: int = 50) -> None:
    """
    Uruchom serię eksperymentów:
    - wczytuje instancje z instance_path (*.json/*.jsonl),
    - stosuje subsetowanie zgodnie z params.subset,
    - dla każdej instancji uruchamia `params.runs` przebiegów,
    - zapisuje każdy wynik do out_path jako 1 linia JSON.
    """
    # Ustalamy listę seedów na runs:
    # - jeśli seeds jest krótsze niż runs -> uzupełniamy 0.. (powtarzalnie)
    seeds = list(params.seeds)
    if len(seeds) < params.runs:
        # proste dopełnienie deterministyczne
        seeds = seeds + list(range(len(seeds), params.runs))

    # Iterujemy po instancjach strumieniowo
    for inst in iter_instances(instance_path):
        inst2 = apply_subset(inst, params.subset)

        # Każdą instancję uruchamiamy `runs` razy
        for r in range(params.runs):
            seed = int(seeds[r])
            if console:
              console.print(f"\n\n[bold green][START][/bold green] [[yellow]Instance[/yellow]: [white]{inst2.meta.get('name','?')}[/white]] [[yellow]Run[/yellow]: [white]{r+1}/{params.runs}[/white]] [[yellow]Seed[/yellow]: [white]{seed}[/white]] [[yellow]n[/yellow]=[white]{len(inst2.items)}[/white]]")     # type: ignore
              line = "="*120
              console.print(f"[white]{line}[/white]")
            else:
              print(f"\n\nStart instance={inst2.meta.get('name','?')} run={r+1}/{params.runs} seed={seed} n={len(inst2.items)}")                          # type: ignore

            run_dict = run_single_ga(inst2, params, seed, time_limit_sec, log_every, run_index=r)

            # Dodatkowe pola identyfikacyjne „run id”
            run_dict["run_index"] = r

            # Zapis jako JSONL (jedna linia)
            write_run_result(run_dict, out_path)