PY=python3
VENV=.venv
ACT=source $(VENV)/bin/activate

install:
\t$(PY) -m venv $(VENV) && $(ACT) && pip install --upgrade pip && pip install -r requirements.txt

run-ui:
\t$(ACT) && streamlit run app/ui_streamlit.py

eval-golden:
\t$(ACT) && $(PY) tools/eval_batch.py --clips data/golden --report export/logs/golden.json

prep-synth:
\t$(ACT) && $(PY) tools/synth_gen.py --out data/synth --n 300

format:
\t$(ACT) && python - <<'PY'\nimport os, pathlib\nprint('No formatter pinned. Use ruff/black if you like.')\nPY
