# GLOBALS
SHELL := /bin/bash
PYTHON_INTEPRETER = "python3"
ENVIRONMENT_NAME = ".env"
KAGGLE_CONFIG = "/content/gdrive/MyDrive/Personal/.kaggle"

# Set .gitignore

# Python environment
env: requirements.txt
	### Creating venv...
	$(PYTHON_INTEPRETER) -m venv $(ENVIRONMENT_NAME)
	### Installing required pkgs...
	$(ENVIRONMENT_NAME)/bin/python -m pip install -r requirements.txt
	### Done!

# Download raw data
datasets: env
	# meh
