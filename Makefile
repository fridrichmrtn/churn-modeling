# GLOBALS
SHELL := /bin/bash
PYTHON_INTEPRETER = "python3"
ENVIRONMENT_NAME = "py-env"
KAGGLE_CONFIG = "/content/gdrive/MyDrive/Personal/.kaggle"

# Set .gitignore

# Python environment
env:
	### Creating venv...
	$(PYTHON_INTEPRETER) -m venv $(ENVIRONMENT_NAME)
	### Installing required pkgs...
	source $(ENVIRONMENT_NAME)/bin/activate
	$(PYTHON_INTEPRETER) -m pip install -r requirements.txt
	### Done!

# Download raw data
datasets:
	### Acquiring Olist dataset...
	kaggle datasets download olistbr/brazilian-ecommerce -p data/raw/olist --unzip
	### Done!
	### Acquiring Retail Rocket dataset...
	kaggle datasets download retailrocket/ecommerce-dataset -p data/raw/rr --unzip
	### Done!
