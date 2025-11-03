"""
Plik: src/runner.py

Cel i rola w projekcie
----------------------
„Klej” wysokiego poziomu:
- strumieniowo wczytuje instancje z `io.iter_instances(...)`,
- stosuje subset (`io.apply_subset(...)`),
- dla każdej instancji wykonuje `Params.runs` uruchomień z różnymi seedami,
- dla każdego uruchomienia:
  - przygotowuje RNG,
  - pętlę wielopokoleniową (wywołując funkcje z `ga.py` i `fitness.py`),
  - zbiera metryki (best/avg per generacja, czas, gen_reached),
  - zapisuje wynik przez `io.write_run_result(...)` (1 linia JSON na uruchomienie).

Interfejs:
- `run_experiment(instance_path: Path, params: Params, out_path: Path) -> None`
   - jedyna publiczna funkcja wołana przez `cli.py`.

Uwagi:
- Tu umieścimy logikę early-stop (patience/min_delta),
- Tu powstanie ślad przebiegu (`trace`) zgodnie z `Params.trace`.
"""