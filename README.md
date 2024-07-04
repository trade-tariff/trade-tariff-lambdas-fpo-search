[![CircleCI](https://dl.circleci.com/status-badge/img/gh/trade-tariff/trade-tariff-lambdas-fpo-search/tree/main.svg?style=svg&circle-token=e0c6d3b2325ad0861a88adbf841eb44ff7b4267a)](https://dl.circleci.com/status-badge/redirect/gh/trade-tariff/trade-tariff-lambdas-fpo-search/tree/main)

# FPO Parcel Item Categorisation API

## Create your development environment

> Make sure you install and enable all pre-commit hooks https://pre-commit.com/

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

In order to do anything useful to run the training, you will need some source
data. This is not included within this repository and will need to be obtained
by contacting the OTT lead.

Once you have obtained the source data for testing, put it in a directory in the
project root named `raw_source_data`.

Once you have the data you can run the training:

```
python train.py
```

To get usage instructions you can run:

```
python train.py --help
```

```
usage: train.py [-h] [--digits DIGITS] [--limit LIMIT] [--learning-rate LEARNING_RATE] [--max-epochs MAX_EPOCHS] [--batch-size BATCH_SIZE] [--device {auto,cpu,mps,cuda}]
                [--embedding-batch-size EMBEDDING_BATCH_SIZE] [--embedding-cache-checkpoint EMBEDDING_CACHE_CHECKPOINT]

Train an FPO classification model.

options:
  -h, --help            show this help message and exit
  --digits DIGITS       how many digits to train the model to
  --limit LIMIT         limit the training data to this many entries to speed up development testing
  --learning-rate LEARNING_RATE
                        the learning rate to train the network with
  --max-epochs MAX_EPOCHS
                        the maximum number of epochs to train the network for
  --batch-size BATCH_SIZE
                        the size of the batches to use when training the model. You should increase this if your GPU has tonnes of RAM!
  --device {auto,cpu,mps,cuda}
                        the torch device to use for training. 'auto' will try to select the best device available.
  --embedding-batch-size EMBEDDING_BATCH_SIZE
                        the size of the batches to use when calculating embeddings. You should increase this if your GPU has tonnes of RAM!
  --embedding-cache-checkpoint EMBEDDING_CACHE_CHECKPOINT
                        how often to update the cached embeddings.
```

#### Notes on data sources

There are two options on a Data Source to control how the codes and descriptions are processed:

##### `creates_codes` to determine what happens with unknown codes

While the data sources are being processed, the `creates_codes` parameter determines what happens when a new code is encountered.

If a new code (i.e. one that hasn't been seen yet) is encountered from a data source and `creates_codes` is `True` for that source then that new code will be added to the set of commodity codes.

If a new code is encountered in a data source and `creates_codes` is `False` for that source then that entry will be skipped.

##### `authoritative` to allow overriding descriptions

While the data sources are being processed, the `authoritative` parameter determines whether a source contains 'definitive' codes for a certain description.

A code for a certain description within an 'authoritative' data source will always override the code for that same description from a non-authoritative source.

For example:

If an authoritative source maps `"widgets"` = `123456`

but then a non-authoritative source also has an entry of `"widgets"` = `098765`, then the non-authoritative entry will be overridden with the authoritative one, so the training data will end up with two entries with the same mapping of:

`"widgets"` = `123456` and `"widgets"` = `123456`

### Benchmarking the model

Once you have trained the model, you can benchmark its performance against some
benchmarking data. You can get some example benchmarking data by contacting the
OTT lead.

These should be `csv` files, with the first column as the first column as the
Goods Description and the second column as the Commodity Code. The first row
should be the header and will be skipped.

Once you have obtained the source data for testing, put it in a directory in the
project root named `benchmarking_data`.

Once you have the data you can run the training:

```
python benchmark.py
```

To get usage instructions you can run:

```
python benchmark.py --help
```

```
usage: benchmark.py [-h] [--digits {2,4,6,8}] [--output {text,json}] [--no-progress] [--colour]

Benchmark an FPO classification model.

options:
  -h, --help            show this help message and exit
  --digits {2,4,6,8}    how many digits to classify the answer to
  --output {text,json}  choose how you want the results outputted
  --no-progress         don't show a progress bar
  --colour              enable ANSI colour for the 'text' output type
```

### Inference

Once you have the model built you can run inference against it to classify
items. By default the inference script requires the following files to be
present:

- `target/subheading.pkl` which is a pcikle file of a list of subheadings. This
  is used to convert the classification from the model back into the eight digit
  code.
- `target/model.pt` which is the PyTorch model

You can either use the training to create fresh versions of these files, or you
can use the pre-built ones. Contact the team lead to get access to them.

#### From the command-line

To get usage instructions you can run:

```
python infer.py --help
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
[61159500 = 225.24, 61159699 = 181.72, 61159900 = 119.44, 61159400 = 71.33]
```

## Licence

FPO Parcel Item Categorisation API is licenced under the [MIT licence](LICENCE.txt)
