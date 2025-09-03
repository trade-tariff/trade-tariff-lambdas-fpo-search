# FPO Parcel Item Categorisation API

## Create your development environment

> Make sure you install and enable all pre-commit hooks https://pre-commit.com/

### Activate the virtual environment (each time the tool is used)

- Create the virtual environment: `python -m venv venv`
- On MacOS or Linux: `source venv/bin/activate`
- On Windows: `.\venv\Scripts\activate`

### Install dependencies

- `pip install -r requirements.txt`

## Usage

### Training the model

In order to do anything useful to run the training, you will need some source
data. This is not included within this repository and will need to be obtained
by contacting the OTT lead.

Once you have obtained the source data for testing, put it in a directory in the
project root named `raw_source_data`.

Once you have the data you can run the training:

```
make train
```

To get usage instructions you can run:

```
python train.py --help
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

These should be `csv` files, with the first column as the Goods Description and
the second column as the Commodity Code. The first row should be the header and will be skipped.

Once you have obtained the source data for testing, put it in a directory in the
project root named `benchmarking_data`.

Once you have the data you can run the training:

```
make benchmark
```

To get usage instructions you can run:

```
python benchmark.py --help
```

### Inference

Once you have the model built you can run inference against it to classify
items. By default the inference script requires the following files to be
present:

- `target/subheading.pkl` which is a pickle file of a list of subheadings. This
  is used to convert the classification from the model back into the eight digit
  code.
- `target/model.pt` which is the PyTorch model
- `target/model_quantized.pt` which is the a much smaller quantized version of the PyTorch model

You can either use the training to create fresh versions of these files, or you
can use the pre-built ones. Contact the team lead to get access to them.

#### From the command-line

To get usage instructions you can run:

```
python infer.py --help
```

For example:

```
python infer.py \
        --query "trousers:62046239" \
        --query "lipstick:33041000" \
        --query "water cup:39241000" \
        --query "towel:63026000" \
        --query "Plenty kitchen towels:48030090" \
        --query "Kingsmill bread:19059030" \
        --digits 8
```

Produces:

```
Query: trousers -> Expected: 62046239 -> Match: True -> Result: [62046239 = 976.44]
Query: lipstick -> Expected: 33041000 -> Match: True -> Result: [33041000 = 913.57]
Query: water cup -> Expected: 39241000 -> Match: True -> Result: [39241000 = 386.11, 39269097 = 54.34, 39249000 = 45.77, 73239300 = 37.19, 96170000 = 36.95]
Query: towel -> Expected: 63026000 -> Match: True -> Result: [63026000 = 896.32]
Query: Plenty kitchen towels -> Expected: 48030090 -> Match: True -> Result: [63026000 = 247.76, 63029100 = 95.38, 63029990 = 54.03, 63029910 = 52.20, 48030090 = 44.81]
Query: Kingsmill bread -> Expected: 19059030 -> Match: True -> Result: [19059030 = 718.73, 19059080 = 116.50]
[61159500 = 225.24, 61159699 = 181.72, 61159900 = 119.44, 61159400 = 71.33]
```

You can forego the expected value by omitting the colon and everything after it:

```
python infer.py \
        --query "trousers" \
        --query "lipstick" \
        --query "water cup" \
        --query "towel" \
        --query "Plenty kitchen towels" \
        --query "Kingsmill bread" \
        --digits 8
```

## Licence

FPO Parcel Item Categorisation API is licenced under the [MIT licence](LICENCE.txt)
