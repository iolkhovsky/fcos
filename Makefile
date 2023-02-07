tests: FORCE
	pytest tests/
FORCE:

install:
	python3 -m venv venv
	source venv/bin/activate
	pip3 install -r requirements.txt

train:
	python3 train.py --config=configs/train.yaml

run:
	python3 run.py --config=configs/run.yaml
