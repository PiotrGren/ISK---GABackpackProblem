> ## Dry runs
#### Tylko weryfikacja I/O + config bez uruchamiania GA

1. **Przykład 1**: Uruchomienie w domyślnymi parametrami:
```bash
python -m src.cli --dry-run
```

2. **Przykład 2**: Pojedyncza instancja + bazowy config
```bash
python -m src.cli \
  --instance data/instances/big-12-proportional-noise-n5000.json \
  --config experiments/configs/base.json \
  --out experiments/results/base.jsonl \
  --dry-run
```

3. **Przykład 3**: Wiele instancji na raz z **.jsonl** - wszystkie 12
```bash
python -m src.cli \
  --instance data/instances/all_big_instances.jsonl \
  --config experiments/configs/base.json \
  --out experiments/results/all_base.jsonl \
  --dry-run
```

4. **Przykład 4**: Nadpisanie kilku parametrów z CLI
```bash
python -m src.cli \
  --instance data/instances/big-07-clustered-n2800.json \
  --config experiments/configs/base.json \
  --out experiments/results/clustered_tuned.jsonl \
  --pop 600 --max-generations 1000 --selection-type tournament --selection-k 5 \
  --dry-run
```

5. **Przykład 5**: Włączenie próbkowania podzbioru (szybki test na 1500 przedmiotach)
```bash
python -m src.cli \
  --instance data/instances/big-12-proportional-noise-n5000.json \
  --config experiments/configs/base.json \
  --out experiments/results/subset_test.jsonl \
  --subset-mode random --subset-size 1500 --subset-seed 999 \
  --dry-run
```

6. **Przykład 6**: Zmiana trybu ograniczeń na karę + ustawienie lambdy + własne seedy
```bash
python -m src.cli \
  --instance data/instances/big-09-heavy-sparse-n3500.json \
  --config experiments/configs/base.json \
  --out experiments/results/penalty_lambda.jsonl \
  --constraint-mode penalty --lambda 25.0 \
  --seeds-csv "0,123,999,2025" \
  --dry-run
```