import argparse
import torch
import tomllib


def config_from_file(func):
    def wrapped(self, *args, **kwargs):
        config_key = func.__name__

        if (
            hasattr(self, "parsed_config")
            and self.parsed_config
            and config_key in self.parsed_config
        ):
            return self.parsed_config[config_key]

        return func(self, *args, **kwargs)

    return wrapped


class TrainScriptArgsParser:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Train an FPO classification model."
        )
        parser.add_argument(
            "--config",
            type=str,
            help="the path to the configuration file to use for training (e.g. search-config.toml)",
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
            "--batch-size",
            type=int,
            help="the size of the batches to use when training the model. You should increase this if your GPU has tonnes of RAM!",
            default=1000,
        )
        parser.add_argument(
            "--device",
            type=str,
            help="the torch device to use for training. if your hardware does not support the device, it will fall back to cpu.",
            choices=["cpu", "mps", "cuda"],
            default="cpu",
        )
        parser.add_argument(
            "--embedding-batch-size",
            type=int,
            help="the size of the batches to use when calculating embeddings. You should increase this if your GPU has tonnes of RAM!",
            default=100,
        )
        parser.add_argument(
            "--embedding-cache-checkpoint",
            type=int,
            help="how often to update the cached embeddings.",
            default=50000,
        )
        parser.add_argument(
            "--vague-terms-data-file",
            type=str,
            help="the path to the vague terms data file",
            default="reference_data/vague_terms.csv",
        )
        parser.add_argument(
            "--extra-references-data-file",
            type=str,
            help="the path to the extra references data file",
            default="reference_data/extra_references.csv",
        )
        parser.add_argument(
            "--cn-data-file",
            type=str,
            help="the path to the CN data file",
            default="reference_data/CN2024_SelfText_EN_DE_FR.csv",
        )
        parser.add_argument(
            "--tradesets-data-dir",
            type=str,
            help="the path to the tradesets data directory",
            default="raw_source_data/tradesets_descriptions",
        )

        self.parsed_args = parser.parse_args()
        self._parse_search_config()


    def torch_device(self):
        arg_device = self.device()

        if arg_device == "cuda" and not torch.cuda.is_available():
            return "cpu"

        if arg_device == "mps" and not torch.backends.mps.is_available():
            return "cpu"

        return arg_device

    @config_from_file
    def device(self):
        return self.parsed_args.device

    @config_from_file
    def learning_rate(self):
        return self.parsed_args.learning_rate

    @config_from_file
    def max_epochs(self):
        return self.parsed_args.max_epochs

    @config_from_file
    def batch_size(self):
        return self.parsed_args.batch_size

    @config_from_file
    def embedding_batch_size(self):
        return self.parsed_args.embedding_batch_size

    @config_from_file
    def embedding_cache_checkpoint(self):
        return self.parsed_args.embedding_cache_checkpoint

    @config_from_file
    def vague_terms_data_file(self):
        return self.parsed_args.vague_terms_data_file

    @config_from_file
    def limit(self):
        return self.parsed_args.limit

    @config_from_file
    def digits(self):
        return self.parsed_args.digits

    @config_from_file
    def extra_references_data_file(self):
        return self.parsed_args.extra_references_data_file

    @config_from_file
    def cn_data_file(self):
        return self.parsed_args.cn_data_file

    @config_from_file
    def tradesets_data_dir(self):
        return self.parsed_args.tradesets_data_dir

    def _parse_search_config(self):
        if self.parsed_args.config is not None:
            with open(self.parsed_args.config, "rb") as f:
                self.parsed_config = tomllib.load(f)
        else:
            self.parsed_config = None
