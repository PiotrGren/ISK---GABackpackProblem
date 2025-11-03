"""
Plik: src/fitness.py

Cel i rola w projekcie
----------------------
Zapewnia funkcje obliczeniowe związane z celem i ograniczeniem:
- `total_weight(bits, items)` i `total_value(bits, items)` - szybkie sumy,
- `fitness_penalty(bits, instance, lambda_)` - kara za przekroczenie pojemności,
- `repair_solution(bits, instance)` - procedura naprawcza (usuwanie przedmiotów
  o najmniejszym value/weight do osiągnięcia feasibility),
- funkcje pomocnicze do wektorowego liczenia fitnessu całej populacji.

Jak łączy się z resztą:
- używane przez `ga.py` podczas ewolucji,
- niezależne od I/O; operują na strukturach z `model.py`.

Uwagi:
- Warto zapewnić wersje *batched* (dla wielu osobników) dla wydajności,
- Dbać o deterministykę - sortowanie stabilne przy jednakowych stosunkach.
"""