.PHONY: install test lint small-games comparison dry-runs results notebooks

install:  ## the package with PyTorch, plus what the notebooks and the tests need
	pip install -e ".[train,notebooks,dev]"

test:
	pytest -q

lint:
	ruff check src tests tools

small-games:  ## train the 20 players of the small games stored in results/ (about 35 minutes on two cores)
	python tools/run_small_games.py

comparison:  ## the two networks on Qubic positions with exact answers (about 10 minutes)
	python tools/compare_networks.py

dry-runs:  ## short self-play runs on Qubic with both networks (about 25 minutes)
	python tools/run_qubic_dry_run.py

results: small-games comparison dry-runs

notebooks:  ## run every notebook again from a clean kernel, in place (a few minutes; reads results/)
	for nb in notebooks/*.ipynb; do \
		jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 $$nb || exit 1; \
	done
