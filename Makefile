tests: FORCE
	pytest tests/
FORCE:

install: FORCE
	python3 -m venv venv
	source venv/bin/activate
	pip3 install -r requirements.txt
FORCE:

install_colab: FORCE
	pip3 install -r requirements_colab.txt
FORCE:

train: FORCE
	python3 train.py --config=configs/train.yaml
FORCE:

train_colab: FORCE
	python train.pu --config=${TRAINING_CONFIG}
FORCE:

train_memprof: FORCE
	python -m memory_profiler train.py --config=configs/train.yaml
FORCE:

run: FORCE
	python3 run.py --config=configs/run.yaml
FORCE:
