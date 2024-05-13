import argparse
from pathlib import Path
import torch
import tomllib
import logging

try:
    # torch_xla is only available when running in EC2 so we dynamically import it if we're running in EC2
    import torch_xla.core.xla_model
except ImportError:
    pass


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


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
            help="the path to the configuration file to use for training (e.g. search-config.toml). Either this or specific arguments must be provided.",
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
            "--model-batch-size",
            type=int,
            help="the size of the batches to use when training the model. You should increase this if your GPU has tonnes of RAM!",
            default=1000,
        )
        parser.add_argument(
            "--device",
            type=str,
            help="the torch device to use for training. if your hardware does not support the device, it will fall back to cpu. auto picks the best device available.",
            choices=["xla", "cuda", "mps", "cpu", "auto"],
            default="auto",
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
        parser.add_argument(
            "--embeddings-cache-enabled",
            type=bool,
            help="whether to cache embeddings or not",
            default=True,
        )

        self.parsed_args = parser.parse_args()
        self._parse_search_config()

    def print(self):
        logger.info("Configuration:")
        logger.info(f"  device: {self.device()}")
        logger.info(f"  learning_rate: {self.learning_rate()}")
        logger.info(f"  max_epochs: {self.max_epochs()}")
        logger.info(f"  model_batch_size: {self.model_batch_size()}")
        logger.info(f"  embedding_batch_size: {self.embedding_batch_size()}")
        logger.info(
            f"  embedding_cache_checkpoint: {self.embedding_cache_checkpoint()}"
        )
        logger.info(f"  vague_terms_data_file: {self.vague_terms_data_file()}")
        logger.info(f"  limit: {self.limit()}")
        logger.info(f"  digits: {self.digits()}")
        logger.info(
            f"  extra_references_data_file: {self.extra_references_data_file()}"
        )
        logger.info(f"  cn_data_file: {self.cn_data_file()}")
        logger.info(f"  tradesets_data_dir: {self.tradesets_data_dir()}")
        logger.info(f"  embeddings_cache_enabled: {self.embeddings_cache_enabled()}")
        logger.info(f"  cache_dir: {self.cache_dir()}")
        logger.info(f"  data_dir: {self.data_dir()}")
        logger.info(f"  target_dir: {self.target_dir()}")
        logger.info(f"  torch_device: {self.torch_device()}")
        logger.info(f"  torch_version: {torch.__version__}")

    def torch_device(self):
        arg_device = self.device()

        if arg_device == "xla":
            return self._xla_device()

        if arg_device == "cuda":
            return self._cuda_device()

        if arg_device == "mps":
            return self._mps_device()

        if arg_device == "auto":
            return self._auto_device()

        return arg_device

    def target_dir(self):
        cwd = Path(__file__).resolve().parent

        return cwd / "target"

    def data_dir(self):
        return self.target_dir() / "training_data"

    def cache_dir(self):
        if self.embeddings_cache_enabled():
            return self.data_dir()
        else:
            return None

    @config_from_file
    def embeddings_cache_enabled(self):
        return self.parsed_args.embeddings_cache_enabled

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
    def model_batch_size(self):
        return self.parsed_args.model_batch_size

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

    def _auto_device(self):
        try:
            if torch_xla.core.xla_model.xla_device_is_available():
                return "xla"
        except NameError:
            pass

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _xla_device(self):
        try:
            if not torch_xla.core.xla_model.xla_device_is_available():
                return "cpu"
        except NameError:
            return "cpu"

        return "xla"

    def _cuda_device(self):
        if not torch.cuda.is_available():
            return "cpu"

        return "cuda"

    def _mps_device(self):
        if not torch.backends.mps.is_available():
            return "cpu"

        return "mps"
