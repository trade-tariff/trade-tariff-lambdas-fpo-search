import argparse
import torch


class FPOArgsParser:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Train an FPO classification model."
        )
        parser.add_argument(
            "--digits",
            type=int,
            help="how many digits to train the model to",
            default=8,
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="limit the training data to this many entries to speed up development testing",
            required=False,
        )
        parser.add_argument(
            "--force",
            help="force the regeneration of source data and embeddings",
            required=False,
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--learning-rate",
            dest="learning_rate",
            type=float,
            help="the learning rate to train the network with",
            default=0.001,
        )
        parser.add_argument(
            "--max-epochs",
            dest="max_epochs",
            type=int,
            help="the maximum number of epochs to train the network for",
            default=3,
        )
        parser.add_argument(
            "--device",
            type=str,
            help="the torch device to use for training. 'auto' will try to select the best device available.",
            choices=["auto", "cpu", "mps", "cuda"],
            default="auto",
        )

        self.parsed_args = parser.parse_args()

    def torch_device(self):
        arg_device = self.parsed_args.device

        if arg_device != "auto":
            return arg_device

        # When "auto" select the best device available
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"
