PYTHON=.venv\Scripts\python.exe
PIP=.venv\Scripts\pip.exe

.PHONY: setup test mypy lint run clean

setup:
	python -m venv .venv
	.venv\Scripts\python.exe -m pip install --upgrade pip
	.venv\Scripts\pip.exe install -r requirements.txt
	.venv\Scripts\pip.exe install -r requirements-dev.txt

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

mypy:
	$(PYTHON) -m mypy .

lint:
	@echo "No explicit linter configured. Run mypy or pytest for validation."

run:
	$(PYTHON) main.py

pycompile:
	$(PYTHON) -m compileall -q -x ".venv" .

clean:
	rmdir /s /q .venv
	del /q *.pyc
	rmdir /s /q __pycache__
