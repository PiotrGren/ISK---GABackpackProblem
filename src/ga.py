"""
Plik: src/ga.py

Cel i rola w projekcie
----------------------
Ten moduł implementuje „silnik” algorytmu genetycznego dla problemu plecakowego 0/1:
- reprezentacja chromosomu: wektor 0/1 (liczby, nie bool) długości n,
- inicjalizacja populacji,
- selekcja rodziców (turniejowa lub ruletka),
- krzyżowanie (one-point oraz uniform),
- mutacja bit-flip z prawdopodobieństwem pm,
- elityzm (przeniesienie najlepszych osobników do następnego pokolenia).

Jak łączy się z resztą:
- `fitness.py` dostarcza funkcję `evaluate_population(...)`, która zwraca fitness,
  oraz (w trybie repair) może zmodyfikować populację, aby była feasible.
- `runner.py` wywołuje funkcje z tego pliku w pętli po generacjach:
  - na wejściu: populacja, fitness, parametry, RNG,
  - na wyjściu: nowe pokolenie.

Założenia:
- Populacja jest przechowywana jako macierz `np.ndarray` o kształcie (P, n)
  z dtype=np.int8 i wartościami {0,1}.
- RNG jest obiektem `numpy.random.Generator` przekazanym z `runner.py`,
  co gwarantuje powtarzalność eksperymentów.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .model import Params



# --- Parametry mutacji ----------------------------------------------------------------------------------
def resolve_pm(pm: str | float, n_items: int) -> float:
  """Zamień pm z configu na liczbę float (np. `1/n` -> 1/n)"""
  if isinstance(pm, str):
    if pm.strip() == "1/n":
      return 1.0 / max(1, n_items)
    raise ValueError(f"Nieznany zapis pm: {pm}")
  return float(pm)



# --- Inicjalizacja populacji ----------------------------------------------------------------------------
def init_population(pop_size: int, n_items: int, rng: np.random.Generator) -> np.ndarray:
  """
  Stwórz populację startową (P, n) jako 0/1 (np.int8)
  Uwaga: startowo losujemy niezależnie bity ~Bernoulli(0.5)
  """
  pop = rng.integers(0, 2, size=(pop_size, n_items), dtype=np.int8)
  return pop



# --- Selekcja --------------------------------------------------------------------------------------------
def tournament_select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
  """
  Wybierz indeks jednego rodzica metodą turniejową.
   - losujemy k kandydatów
   - wybieramy tego o największym fitness
  """
  idx = rng.integers(0, fitness.shape[0], size=k)
  
  best = idx[np.argmax(fitness[idx])]
  return int(best)


def roulette_select(fitness: np.ndarray, rng: np.random.Generator) -> int:
  """
  Wybierz indeks jednego rodzica metodą ruletki.
  Uwaga: ruletka wymaga nieujemnych wag - dlatego przesuwamy fitness do >= 0
  """
  f = fitness.astype(np.float64)
  
  min_f = float(np.min(f))
  if min_f < 0:
    f = f - min_f
  
  total = float(np.sum(f))
  if total <= 0:
    return int(rng.integers(0, fitness.shape[0]))
  
  probs = f / total
  return int(rng.choice(fitness.shape[0], p=probs))


def select_parent(fitness: np.ndarray, params: Params, rng: np.random.Generator) -> int:
  """Wybierz rodzica zgodnie z konfiguracją selekcji"""
  if params.selection.type == "tournament":
    return tournament_select(fitness, params.selection.k, rng)
  if params.selection.type == "roulette":
    return roulette_select(fitness, rng)
  raise ValueError(f"Nieznany typ selekcji: {params.selection.type}")



# --- Krzyżowanie -----------------------------------------------------------------------------------------
def crossover_one_point(p1: np.ndarray, p2: np.ndarray, pc: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
  """
  Krzyżowanie jednopunktowe:
   - z prawdopodobieństwem pc robimy crossover
   - w przeciwnym razie kopiujemy rodziców
  """
  n = p1.shape[0]
  if n < 2 or rng.random() >= pc:
    return p1.copy(), p2.copy()
  
  cut = int(rng.integers(1, n))
  c1 = np.concatenate([p1[:cut], p2[cut:]]).astype(np.int8, copy=False)
  c2 = np.concatenate([p2[:cut], p1[cut:]]).astype(np.int8, copy=False)
  return c1, c2


def crossover_uniform(
    p1: np.ndarray, p2: np.ndarray, pc: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Krzyżowanie uniform:
    - z prawdopodobieństwem pc mieszamy geny maską losową,
    - w przeciwnym razie kopiujemy rodziców.
    """
    n = p1.shape[0]
    if rng.random() >= pc:
        return p1.copy(), p2.copy()

    mask = rng.integers(0, 2, size=n, dtype=np.int8)  # 0/1 maska
    c1 = np.where(mask == 1, p1, p2).astype(np.int8)
    c2 = np.where(mask == 1, p2, p1).astype(np.int8)
    return c1, c2


def crossover(p1: np.ndarray, p2: np.ndarray, params: Params, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Wybierz operator krzyżowania zgodnie z params.crossover."""
    if params.crossover == "one_point":
        return crossover_one_point(p1, p2, params.pc, rng)
    if params.crossover == "uniform":
        return crossover_uniform(p1, p2, params.pc, rng)
    raise ValueError(f"Nieznany operator crossover: {params.crossover}")



# --- Mutacja -------------------------------------------------------------------------------------------------
def mutate_bitflip(child: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
    """
    Mutacja bit-flip:
    - dla każdego genu z prawdopodobieństwem pm odwracamy 0<->1.
    """

    m = rng.random(child.shape[0]) < pm

    child[m] = (1 - child[m]).astype(np.int8, copy=False)
    return child


def mutate(child: np.ndarray, params: Params, pm_value: float, rng: np.random.Generator) -> np.ndarray:
    """Wybierz operator mutacji zgodnie z params.mutation."""
    if params.mutation == "bit_flip":
        return mutate_bitflip(child, pm_value, rng)
    raise ValueError(f"Nieznany operator mutacji: {params.mutation}")



# --- Elityzm --------------------------------------------------------------------------------------------------
def get_elite_indices(fitness: np.ndarray, elitism: int) -> np.ndarray:
    """Zwróć indeksy elit (najlepszych osobników) - największy fitness."""
    if elitism <= 0:
        return np.array([], dtype=np.int64)
    # argsort rosnąco, więc bierzemy ogon
    idx = np.argsort(fitness)[-elitism:]
    return idx.astype(np.int64)


# --- Jedna generacja -------------------------------------------------------------------------------------------

def next_generation(
    pop: np.ndarray,
    fitness: np.ndarray,
    params: Params,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Zbuduj następne pokolenie.
    Uwaga: tu nie liczymy fitnessu (to robi `fitness.evaluate_population`).
    """
    P, n = pop.shape
    pm_value = resolve_pm(params.pm, n)

    new_pop = np.empty_like(pop, dtype=np.int8)

    # 1) Elityzm: kopiujemy najlepszych na początek nowej populacji
    elite_idx = get_elite_indices(fitness, params.elitism)
    e = elite_idx.size
    if e > 0:
        new_pop[:e] = pop[elite_idx]

    # 2) Uzupełniamy resztę przez selekcję + crossover + mutację
    i = e
    while i < P:
        # wybór dwóch rodziców
        p1_idx = select_parent(fitness, params, rng)
        p2_idx = select_parent(fitness, params, rng)
        p1 = pop[p1_idx]
        p2 = pop[p2_idx]

        # crossover -> dwoje dzieci
        c1, c2 = crossover(p1, p2, params, rng)

        # mutacja
        c1 = mutate(c1, params, pm_value, rng)
        c2 = mutate(c2, params, pm_value, rng)

        # zapis do populacji (uważamy na nieparzysty P)
        new_pop[i] = c1
        i += 1
        if i < P:
            new_pop[i] = c2
            i += 1

    return new_pop