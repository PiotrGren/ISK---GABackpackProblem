"""
Plik: src/model.py

Cel i rola w projekcie
----------------------
Zawiera *modele danych* (Pydantic v2) używane w całym projekcie:
- `Item` i `Instance` - reprezentacja wejściowej instancji problemu plecakowego,
- `Params` i pomocnicze konfiguracje (`SelectionConfig`, `ConstraintConfig`,
  `EarlyStopConfig`, `SubsetConfig`, `TraceConfig`) - scalają wszystkie parametry
  algorytmu i uruchomienia w *jednym, walidowanym miejscu*.

Jak łączy się z resztą:
- `src/io.py` korzysta z tych modeli do walidacji danych wczytywanych z plików,
- `src/cli.py` przekształca JSON config w obiekt `Params`, nakłada nadpisania z CLI,
- `src/ga.py` i `src/runner.py` używają `Instance` i `Params` w czasie ewolucji.

Powiązanie z projektem:
- Pola i struktury odpowiadają dokumentacji w README i przykładowemu `base.json`.
- Dzięki Pydantic unikamy cichych błędów typu literówki w nazwach pól w configu.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator



# --- Modele instancji -------------------------------------------------------------------------
class Item(BaseModel):
    """Pojedynczy przedmiot w plecaku"""
    id: int
    weight: float = Field(gt=0, alias="weight", description="Waga przedmiotu (>0)")
    value: float = Field(ge=0, description="Wartośćprzedmiotu (>=0)")
    

class Instance(BaseModel):
    """Instancja problemu: pojemność + lista przedmiotów + metadane"""
    capacity: float = Field(gt=0, description="Pojemność plecaka (>0)")
    items: List[Item] = Field(default_factory=list)
    meta: Optional[dict[str, Any]] = None
    
    @property
    def n_items(self) -> int:
        """Zwraca liczbę przedmotów w instancji"""
        return len(self.items)
    
    
    
# --- Konfiguracja parametrów -------------------------------------------------------------------
class SelectionConfig(BaseModel):
    """Konfiguracja selekcji (np. turniejowa)"""
    type: Literal["tournament", "roulette"] = "tournament"
    k: int = Field(3, ge=2, description="Rozmiar turnieju (dla tournament tylko)")
    
    
class ConstraintConfig(BaseModel):
    """Sposób obsługi ograniczenia pojemności: naprawa lub kara"""
    mode: Literal["repair", "penalty"] = "repair"
    lambda_: float = Field(10.0, alias="lambda", ge=0, description="Współczynnik kary (tylko dla penalty)")
    
    model_config = dict(populate_by_name=True) # type: ignore
    
    
class EarlyStopConfig(BaseModel):
    """Paramter wczesnego stopu"""
    patience: int = Field(0, ge=0)
    min_delta: float = 0.0
    
    
class SubsetConfig(BaseModel):
    """Opcjonalne przycięcie liczby przedmiotów (do testów / szybkich przebiegów)"""
    mode: Literal["none", "random", "first_k"] = "none"
    size: int = Field(0, ge=0, description="Maksymalna liczba przedmiotów po przycięciu")
    seed: int = 0
    
    
class TraceConfig(BaseModel):
    """Co logować w śladzie przebiegu"""
    store_best_per_gen: bool = True
    store_avg_per_gen: bool = True
    

class Params(BaseModel):
    """Główny zbiór parmetrów algorytmu i uruchonmienia"""
    population: int = Field(..., ge=2)
    pc: float = Field(..., ge=0.0, le=1.0, description="Prawdopodobieństwo krzyżowania")
    pm: str | float = Field(..., description='Np. "1/n" albo stała 0.01')
    elitism: int = Field(0, ge=0)

    selection: SelectionConfig = Field(default_factory=SelectionConfig)             # type: ignore
    crossover: Literal["one_point", "uniform"] = "one_point"
    mutation: Literal["bit_flip"] = "bit_flip"
    constraint: ConstraintConfig = Field(default_factory=ConstraintConfig)          # type: ignore

    max_generations: int = Field(..., ge=1)
    early_stop: EarlyStopConfig = Field(default_factory=EarlyStopConfig)            # type: ignore

    runs: int = Field(1, ge=1)
    seeds: List[int] = Field(default_factory=lambda: [0])

    subset: SubsetConfig = Field(default_factory=SubsetConfig)                      # type: ignore
    trace: TraceConfig = Field(default_factory=TraceConfig)

    @field_validator("pm")
    @classmethod
    def _validate_pm(cls, v):
        """Dopuszczamy '1/n' oraz liczby z zakresu [0,1]."""
        if isinstance(v, (int, float)):
            if 0 <= float(v) <= 1:
                return float(v)
            raise ValueError("pm jako liczba musi być w [0,1]")
        if isinstance(v, str) and v.strip() == "1/n":
            return v.strip()
        raise ValueError('pm musi być "1/n" lub liczbą z [0,1]')