[![CircleCI](https://dl.circleci.com/status-badge/img/gh/trade-tariff/trade-tariff-lambdas-fpo-search/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/trade-tariff/trade-tariff-lambdas-fpo-search/tree/main)

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
% python train.py --help
```
```
usage: train.py [-h] [--digits DIGITS] [--limit LIMIT] [--force] [--learning-rate LEARNING_RATE] [--max-epochs MAX_EPOCHS] [--device {auto,cpu,mps,cuda}]

Train an FPO classification model.

options:
  -h, --help            show this help message and exit
  --digits DIGITS       how many digits to train the model to
  --limit LIMIT         limit the training data to this many entries to speed up development testing
  --force               force the regeneration of source data and embeddings
  --learning-rate LEARNING_RATE
                        the learning rate to train the network with
  --max-epochs MAX_EPOCHS
                        the maximum number of epochs to train the network for
  --device {auto,cpu,mps,cuda}
                        the torch device to use for training. 'auto' will try to select the best device available.
```

### Inference

Once you have the model built you can run inference against it to classify items. By default the inference script requires the following files to be present:

- `target/subheading.pkl` which is a pcikle file of a list of subheadings. This is used to convert the classification from the model back into the eight digit code.
- `target/model.pt` which is the PyTorch model

You can either use the training to create fresh versions of these files, or you can use the pre-built ones. Contact the team lead to get access to them.

#### From the command-line

To get usage instructions you can run:
```
% python infer.py --help
```
```
usage: infer.py [-h] [--limit LIMIT] query

Query an FPO classification model.

positional arguments:
  query          the query string

options:
  -h, --help     show this help message and exit
  --limit LIMIT  limit the number of responses
  --digits {2,4,6,8}  how many digits to classify the answer to
```

For example:

```
python infer.py --limit 10 --digits 8 'smelly socks'
```
```
[61159500 = 225.24, 61159699 = 181.72, 61159900 = 119.44, 61159400 = 71.33, 61151090 = 27.60, 62179000 = 17.30, 61159610 = 17.30, 62171000 = 13.81, 61151010 = 13.68, 62052000 = 7.75]
```

#### Running as an API

Start the FastAPI service:

Either using:

```python api.py```

or

```uvicorn api:app --port 5000```

You can then access the service locally at http://localhost:5000/code-search?q=smelly+socks&limit=10&digits=8

#### Building an inference API Docker image

You can build the API as a Docker image:

```docker build -f Dockerfile.inference -t fpo-inference-api .```

And then run it:

```docker run -p 5000:5000 fpo-inference-api```

You can then access the service locally at http://localhost:5000/code-search?q=smelly+socks&limit=10&digits=8

## Licence

FPO Parcel Item Categorisation API is licenced under the [MIT licence](LICENCE.txt)
