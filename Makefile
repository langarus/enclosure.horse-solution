VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: init run web clean

init: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

run: $(VENV)/bin/activate
	$(PY) solve.py img.png solution.png

web: $(VENV)/bin/activate
	$(PY) app.py

clean:
	rm -rf $(VENV) solution.png
