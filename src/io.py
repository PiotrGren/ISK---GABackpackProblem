"""
Plik: src/io.py

Cel i rola w projekcie
----------------------
Ten moduł odpowiada za *wszystkie operacje wejścia/wyjścia*:
- wczytywanie instancji problemu plecakowego z plików JSON i JSONL,
- strumieniowe odczytywanie wielu instancji (plik .jsonl: 1 linia = 1 instancja),
- bezpieczny zapis wyników pojedynczego uruchomienia GA do plików *.jsonl
  (1 linia = 1 wynik), co pozwala efektywnie gromadzić duże serie eksperymentów,
- opcjonalne, konfigurowalne *próbkowanie podzbioru przedmiotów* (subset), aby
  umożliwić szybkie testy na fragmentach dużych instancji.

Jak łączy się z resztą:
- korzysta z modeli z `src/model.py` (Pydantic) do walidacji struktur danych,
- będzie używany przez `src/cli.py` (CLI), które woła funkcje z tego pliku, aby:
  (1) wczytać instancje,
  (2) opcjonalnie zastosować subset (zgodnie z configiem),
  (3) przekazać gotowe obiekty `Instance` do pętli ewolucji (w `runner.py`),
  (4) dopisywać rezultaty (słowniki) do pliku wyników *.jsonl.

Powiązanie z projektem:
- Format wejścia i wyjścia jest zgodny z ustaleniami z README: instancja JSON
  zawiera `capacity`, tablicę `items[{id, weight, value}]` oraz `meta`.
- Ten moduł jest niezależny od implementacji GA — można go testować oddzielnie.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union
try:
    import orjson as _json
except Exception:
    import json as _json

from .model import Instance, SubsetConfig


# -- JSON utils -------------------------------------------------------------------------------------
def _loads(s: Union[str, bytes]) -> Dict:
    """Parse JSON string/bytes -> dict"""
    return _json.loads(s)

def _dumps(obj: Dict) -> str:
    """Dump dict -> JSON string z wyłączonym ascii-escaping i stabilną kolejnością."""
    if _json.__name__ == "orjson":
        return _json.dumps(obj).decode("utf-8")         # type: ignore
    return _json.dumps(obj, ensure_ascii=False)         # type: ignore



# -- Wczytywanie instancji ---------------------------------------------------------------------------
def read_json(path: Union[str, Path]) -> Dict:
    """Wczytuje plik JSON (pojedyncza instancja) i zwraca jego zawartość jako słownik."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return _loads(text)

def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict]:
    """Iteruj po rekordach pliku JSONL (1 linia = 1 instancja), pomijając puste linie."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield _loads(line)
            
def load_instance_from_dict(d: Dict) -> Instance:
    """Zamień słownik na zwalidowaną instancję `Instance` (Pydantic)."""
    return Instance.model_validate(d)

def load_instance(path: Union[str, Path]) -> Instance:
    """Wczytaj pojedynczą instancję z pliku JSON i zwróć obiekt `Instance`."""
    d = read_json(path)
    return load_instance_from_dict(d)

def iter_instances(path: Union[str, Path]) -> Iterator[Instance]:
    """
    Wczytaj jedną lub wiele instancji:
     - *.json -> dokładnie jedna instancja
     - *.jsonl -> wiele instancji (1 linia = 1 instancja)
    """
    p = Path(path)
    if p.suffix == ".jsonl":
        for rec in iter_jsonl(p):
            yield load_instance_from_dict(rec)
    elif p.suffix == ".json":
        yield load_instance_from_dict(read_json(p))
    else:   # pragma: no cover
        raise ValueError(f"Nieobsługiwany format pliku: {p.suffix}. Obsługiwane: .json, .jsonl")
    


# -- Subsetowanie przedmiotów (wybór tylko kilku z całego zbioru) ---------------------------------------
def apply_subset(inst: Instance, subset: Optional[SubsetConfig]) -> Instance:
    """
    Zastosuj reguły subsetowania (none/random/first_k).
    Zwraca "nową" instancję, a oryginał pozostaje niezmieniony.
    """
    if not subset or subset.mode == "none":
        return inst
    
    n = len(inst.items)
    k = min(subset.size, n)
    if subset.mode =="first_k":
        chosen = inst.items[:k]
    elif subset.mode == "random":
        import random
        
        rng = random.Random(subset.seed)
        indices = list(range(n))
        rng.shuffle(indices)
        picked = indices[:k]
        picked.sort()
        chosen = [inst.items[i] for i in picked]
    else:   # pragma: no cover
        raise ValueError(f"Nieobsługiwany tryb subsetowania: {subset.mode}")
    
    # Zwracamy nową instancję z tymi samymmi capacity/meta ale z przyciętymi items
    data = inst.model_dump()
    data["items"] = [it.model_dump() for it in chosen]
    
    # Notujemy w meta, że zastosowano subsetowanie, dla czytelności w wynikach
    meta = dict(data.get("meta") or {})
    meta["subset_applied"] = True
    meta["subset_mode"] = subset.mode
    meta["subset_size"] = k
    data["meta"] = meta
    
    return Instance.model_validate(data)



# -- Zapis wyników ---------------------------------------------------------------------------------------
def write_run_result(run: Dict, out_path: Union[str, Path]) -> None:
    """Dopisz pojedynczy wynik (dict) jako jedną linię w wynikowym JSONL"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = _dumps(run)
    with p.open("a", encoding='utf-8') as f:
        f.write(line)
        f.write("\n")
        f.close()