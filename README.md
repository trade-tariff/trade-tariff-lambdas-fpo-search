# FPO Parcel Item Categorisation API

## Create your virtual environment
It is recommended that you run the tool within a virtual environment. These instructions use `venv` but you should be able to use whichever you like.

### Create the virtual environment (this is a one-off step)
- Create a virtual environment: `python -m venv venv/`

### Activate the virtual environment (each time the tool is used)
- On MacOS or Linux: `source venv/bin/activate`
- On Windows: `.\venv\Scripts\activate`
- Install necessary Python modules via `pip install -r requirements.txt`

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