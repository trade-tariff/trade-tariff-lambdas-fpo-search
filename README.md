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

### Inference

Once you have the model built you can run inference against it to classify items. By default the inference script requires the following files to be present:

- `target/subheading.pkl` which is a pcikle file of a list of subheadings. This is used to convert the classification from the model back into the eight digit code.
- `target/model.pt` which is the PyTorch model

You can either use the training to create fresh versions of these files, or you can use the pre-built ones. Contact the team lead to get access to them.

#### From the command-line

```
% python infer.py --help
usage: infer.py [-h] [--limit LIMIT] query

Query an FPO classification model.

positional arguments:
  query          the query string

options:
  -h, --help     show this help message and exit
  --limit LIMIT  limit the number of responses

```

For example:

`python infer.py --limit 10 'smelly socks'`

#### Running as an API

Start the FastAPI service:
- Either `python api.py`
- or `uvicorn api:app --port 5000`

You can then access the service locally at http://localhost:5000/search?q=smelly+socks&limit=10


## Licence

FPO Parcel Item Categorisation API is licenced under the [MIT licence](LICENCE.txt)