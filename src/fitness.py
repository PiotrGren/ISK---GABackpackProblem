"""
Plik: src/fitness.py

Cel i rola w projekcie
----------------------
Ten moduł zawiera całą logikę „matematyczną” dla problemu plecakowego 0/1:
- liczenie wagi i wartości rozwiązania (chromosomu) dla pojedynczego osobnika,
- liczenie wagi i wartości *dla całej populacji* (wektorowo w NumPy),
- obsługa ograniczenia pojemności plecaka w dwóch trybach:
  1) penalty  - fitness = value - lambda * max(0, overweight)
  2) repair   - naprawa rozwiązania poprzez zdejmowanie przedmiotów,
               aż rozwiązanie stanie się dopuszczalne (waga <= capacity)

Jak łączy się z innymi plikami:
- `ga.py` wywołuje funkcje z tego pliku, żeby:
  - szybko policzyć fitness populacji,
  - ewentualnie naprawić osobniki (tryb repair),
  - sprawdzać czy rozwiązanie jest feasible (waga <= capacity).
- `runner.py` buduje dane wejściowe do obliczeń:
  - wektory `weights` i `values` na podstawie `Instance.items`,
  - przekazuje je do funkcji fitnessu, aby uniknąć ciągłego dostępu do obiektów.

Założenia / konwencje:
- Chromosom = wektor bitów {0,1} o długości n (n = liczba przedmiotów).
- Wagi i wartości przechowujemy jako `np.ndarray` float (szybko i prosto).
- Funkcje są „czyste” (nie robią I/O). Jedyne „losowe” rzeczy są w GA,
  a nie w fitnessie.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .model import Instance


# --- Pomocnicze: przygotowanie danych -------------------------------------------------------------------
def build_item_arrays(instance: Instance) -> Tuple[np.ndarray, np.ndarray]:
  """Zbuduj wektory weights/values (długości n) z obiektu Instance"""
  w = np.array([it.weight for it in instance.items], dtype=np.float64)
  v = np.array([it.value for it in instance.items], dtype=np.float64)
  return w, v



# --- Suma wagi / wartości dla jednego rozwiązania -------------------------------------------------------
def total_weight(bits: np.ndarray, weights: np.ndarray) -> float:
  """Zwróć łączną wagę rozwiązania (wektor bitów) dla zadanych weights"""
  return float(np.dot(bits, weights))             #bits: (n,) | wagi: (n,)


def total_value(bits:np.ndarray, values: np.ndarray) -> float:
  """Zwróć łączną wartość rozwiązania (wektor bitów) dla zadanych values"""
  return float(np.dot(bits, values))


def is_feasible(bits: np.ndarray, weights: np.ndarray, capacity: float) -> bool:
  """Sprawdź czy rozwiązanie jest dopuszczalne: waga <= capacity"""
  return total_weight(bits, weights) <= capacity



# --- Suma wagi / wartości dla populacji (batched) --------------------------------------------------------
def population_weights(pop: np.ndarray, weights: np.ndarray) -> np.ndarray:
  """Zwróć wektor wag dla populacji: pop.shape=(P,n) -> (P,)"""
  return pop @ weights


def population_values(pop: np.ndarray, values: np.ndarray) -> np.ndarray:
  """Zwróć wektor wartości dla populacji: pop.shape(P,n) -> (P,)"""
  return pop @ values



# --- Penalty fitness -------------------------------------------------------------------------------------
def fitness_penalty(
  pop: np.ndarray,
  weights: np.ndarray,
  values: np.ndarray,
  capacity: float,
  lambda_: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Policz fitness całej populacji w trybie penalty.
  
  Zwraca:
   - fitness: (P,)
   - w_sum:   (P,)
   - v_sum:   (P,)
  """
  w_sum = population_weights(pop, weights)
  v_sum = population_values(pop, values)
  
  overweight = np.maximum(0.0, w_sum - capacity)
  
  fit = v_sum - (lambda_ * overweight)
  return fit, w_sum, v_sum



# --- Repair (naprawa) ----------------------------------------------------------------------------------
def repair_solution(
  bits: np.ndarray,
  weights: np.ndarray,
  values: np.ndarray,
  capacity: float
) -> np.ndarray:
  """
  Napraw rozwiązanie (pojedynczy chromosom) tak, aby stało się feasible.
  
  Strategia:
   - jeśli waga <= capacity -> zwracamy bez zmian
   - jeśli waga > capacity:
     zdejmujemy kolejno przedmioty o najgorszym stosunku value/weight (najmniejszym),
     ALE tylko spośród tych, które są aktualnie w plecaki (bit == 1)
     aż do uzyskania weights <= capacity
  
  Deterministyczność:
   - przy równym value/weight sortujemy stabilnie po indeksie (rosnąco)
  """
  x = bits.copy()
  
  current_w = float(np.dot(x, weights))
  if current_w <= capacity:
    return x
  
  chosen_idx = np.flatnonzero(x == 1)
  if chosen_idx.size == 0:
     return x
   
  ratio = values[chosen_idx] / np.maximum(weights[chosen_idx], 1e-12)
  
  order = np.lexsort((chosen_idx, ratio))
  remove_seq = chosen_idx[order]
  
  for j in remove_seq:
    if current_w <= capacity:
      break
    x[j] = 0
    current_w -= float(weights[j])
    
  return x


def repair_population(
  pop: np.ndarray,
  weights: np.ndarray,
  values: np.ndarray,
  capacity: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Napraw całą populację (P, n) w trybie repair.
  
  Zwraca:
   - pop_fixed (P, n)
   - w_sum     (P,)
   - v_sum     (P,)
  """
  pop_fixed = pop.copy()
  
  w_sum = population_weights(pop_fixed, weights)
  overweight_idx = np.flatnonzero(w_sum > capacity)
  
  for i in overweight_idx:
    pop_fixed[i] = repair_solution(pop_fixed[i], weights, values, capacity)
    
  w_sum2 = population_weights(pop_fixed, weights)
  v_sum2 = population_values(pop_fixed, values)
  
  return pop_fixed, w_sum2, v_sum2



# --- Wspólny interfejs: fitness dla GA ------------------------------------------------------------------------
def evaluate_population(
  pop: np.ndarray,
  weights: np.ndarray,
  values: np.ndarray,
  capacity: float,
  constraint_mode: str,
  lambda_: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Główna funkcja "evaluate" dla GA.
  
  Zwraca: 
   - pop_out: (P,n) populacja po ewentualnej naprawie (dla repair), inaczej ta sama
   - fitness: (P,)
   - w_sum:   (P,)
   - v_sum:   (P,)
  
  Uwaga:
   - W trybie repair fitness = v_sum (bo wszystkie powinny być feasible),
     a jeśli coś nie da się naprawić (teoretycznie), to nadal fitness = v_sum,
     ale wagi pokażą czy jest feasible
  """
  if constraint_mode == "penalty":
    fit, w_sum, v_sum = fitness_penalty(pop, weights, values, capacity, lambda_)
    return pop, fit, w_sum, v_sum
  
  if constraint_mode == "repair":
    pop_fixed, w_sum, v_sum = repair_population(pop, weights, values, capacity)
    fit = v_sum
    return pop_fixed, fit, w_sum, v_sum
  
  raise ValueError(f"Nieznany b constraint_mode: {constraint_mode}. Użyj `repair` albo `penalty`.")