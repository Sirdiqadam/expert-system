PY ?= python
VENV ?= .venv
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif
PYV := $(BIN)/python

.PHONY: setup test demo eval booklets exam exam-kb lint clean

setup: ## Create venv and install the package with dev + ui extras
	$(PY) -m venv $(VENV)
	$(PYV) -m pip install --upgrade pip
	$(PYV) -m pip install -e ".[dev,ui,nlp]"

test: ## Run the test suite
	$(PYV) -m pytest

demo: ## Launch the Streamlit demo app
	$(PYV) -m streamlit run app/streamlit_app.py

eval: ## Run the evaluation harness over the sample set
	$(PYV) -m eval.run_eval

booklets: ## (Re)generate the sample scanned booklets in data/sample_booklets
	$(PYV) tools/make_sample_booklets.py

exam-kb: ## Regenerate the COM 226 KB (rubric/rules) from kb/exams/com226/questions.yaml
	$(PYV) tools/build_exam_kb.py com226

exam: ## Mark the sample COM 226 script (offline unless an API key is set)
	$(PYV) -m ees.exam data/exam/com226/sample_student_script.txt

clean: ## Remove caches and build artifacts
	rm -rf build dist .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
