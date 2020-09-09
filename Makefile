init:
	python3 -m pip install --user pipenv
	pipenv --python 3.7
	pipenv install --dev --skip-lock

LIB_LINT = ROTASK
RUN_LINT = run_*.py
PYTHON_LINT = $(LIB_LINT) $(RUN_LINT)

lint: ending flake8 pylint mypy

linter_version:
	pipenv run pip list | grep -P "(flake8|pyflakes|pycodestyle)"
	pipenv run pip list | grep -P "(pylint|astroid)"
	pipenv run pip list | grep -P "(mypy|typing|typed)"

ending:
	! grep -rHnP --include="*.py" --include="*.json" --include="*.md" --include="*.csv" "\x0D" ${PYTHON_LINT}

flake8:
	pipenv run flake8 ${PYTHON_LINT}

pylint:
	pipenv run pylint ${PYTHON_LINT}

clean_mypy:
	rm -rf .mypy_cache/

mypy: clean_mypy
	pipenv run mypy ${PYTHON_LINT}

clean_log:
	rm ./logs/*.log

# Set WANDB_DIR to prevent "./wandb" from forming
WANDB_DIR = "/tmp/ROTASK_wandb/"
set_sweep:
	WANDB_DIR=$(WANDB_DIR) pipenv run wandb sweep --project "REMOVED" --entity "REMOVED" sweep.yaml
set_ecg_sweep:
	WANDB_DIR=$(WANDB_DIR) pipenv run wandb sweep --project "REMOVED" --entity "REMOVED" ecg_sweep.yaml

# --project <project> --entity <repo> --count <max_run> <sweep-id>
# Set CUDA_VISIBLE_DEVICES so that inside we can all use GPU 0
run_sweep_agent:
	echo "Setting file descriptor soft limit to 102400" && ulimit -n 102400 && ulimit -Sn && \
		CUDA_VISIBLE_DEVICES=7 WANDB_DIR=$(WANDB_DIR) pipenv run wandb agent --project "REMOVED" --entity "REMOVED" --count 50 <sweep-id>

run_experiment:
	echo "Setting file descriptor soft limit to 102400" && ulimit -n 102400 && ulimit -Sn && pipenv run python3 run_experiment.py

run_ecg_experiment:
	echo "Setting file descriptor soft limit to 102400" && ulimit -n 102400 && ulimit -Sn && pipenv run python3 run_ecg_experiment.py

run_ecgfdb_experiment:
	echo "Setting file descriptor soft limit to 102400" && ulimit -n 102400 && ulimit -Sn && pipenv run python3 run_ecgfdb_experiment.py
