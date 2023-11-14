# FPO Parcel Item Categorisation API

## Create your development environment

### Initial development environment setup (one-time setup)

- Set up a developer environment: `make dev-env`

### Activate the virtual environment (each time the tool is used)

- On MacOS or Linux: `source venv/bin/activate`
- On Windows: `.\venv\Scripts\activate`

### Install dependencies when the `pyproject.toml` has changed

- For development: `make install-dev`
- For production: `make install`

## Usage

### Training the model

In order to do anything useful to run the training, you will need some source data. This is not included within this repository and will need to be obtained by contacting the OTT lead.

Once you have obtained the source data for testing, put it in a directory in the project root named `raw_source_data`.

Once you have the data you can run the training:

```
python train.py
```

To get usage instructions you can run:

```
python train.py --help
```
