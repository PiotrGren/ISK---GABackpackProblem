"""
Plik: src/ga.py

Cel i rola w projekcie
----------------------
Zawiera *czyste operatory GA* i logikę tworzącą nowe pokolenia:
- inicjalizacja populacji chromosomów binarnych,
- selekcja (turniejowa/ruletka),
- krzyżowanie (one_point/uniform) z obsługą prawdopodobieństwa pc,
- mutacja (bit_flip) z odpowiednim pm (w tym interpretacja "1/n"),
- elityzm (kopiowanie najlepszych osobników),
- interfejs: funkcja `evolve_one_generation(...)` i/lub `evolve(...)`, które
  nie wiedzą nic o I/O – przyjmują obiekty `Instance`, `Params` i zwracają
  nowe populacje/fitnessy.

Jak łączy się z resztą:
- `fitness.py` dostarcza funkcje liczące wartości i ograniczenia (penalty/repair),
- `runner.py` zawiera pętlę wysokiego poziomu (wielopokoleniowa, logowanie trace),
- `cli.py` tylko konfiguruje i woła `runner`, nie dotyka bezpośrednio `ga.py`.

Uwagi implementacyjne:
- Dopracować wektorowe liczenie fitnessu (numpy) dla szybkości,
- Oddzielić kod „czysty” (deterministyczny przy danym seedzie) od losowań (RNG).
"""