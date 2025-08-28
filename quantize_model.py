import toml
import torch
from torch.quantization import quantize_dynamic

from model.model import SimpleNN
from train_args import TrainScriptArgsParser

args = TrainScriptArgsParser()
args.load_config_file()

device = "cpu"


def load_model():
    model_file = args.target_dir() / "model.pt"
    model_config = toml.load(args.target_dir() / "model.toml")

    model = SimpleNN(
        model_config["input_size"],
        model_config["hidden_size"],
        model_config["output_size"],
        model_config["dropout_layer_1_percentage"],
        model_config["dropout_layer_2_percentage"],
    )

    model.load_state_dict(torch.load(model_file, map_location=device))

    model.eval()

    return model


# NOTE: Apply dynamic quantization to the model


# This will quantize the weights of Linear layers to int8 significantly reducing the model size whilst maintaining accuracy
def quantize_model(model):
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_model, args.target_dir() / "model.pt")


if __name__ == "__main__":
    model = load_model()

    quantize_model(model)
