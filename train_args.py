import argparse
from pathlib import Path
import torch
import toml
import logging

logger = logging.getLogger("config")
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
            required=False,
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
            default=0.0011,
        )
        parser.add_argument(
            "--max-epochs",
            dest="max_epochs",
            type=int,
            help="the maximum number of epochs to train the network for",
            default=4,
        )
        parser.add_argument(
            "--model-batch-size",
            type=int,
            help="the size of the batches to use when training the model. You should increase this if your GPU has tonnes of RAM!",
            default=1024,
        )
        parser.add_argument(
            "--device",
            type=str,
            help="the torch device to use for training. if your hardware does not support the device, it will fall back to cpu. auto picks the best device available.",
            choices=["cuda", "mps", "cpu", "auto"],
            default="auto",
        )
        parser.add_argument(
            "--embedding-batch-size",
            type=int,
            help="the size of the batches to use when calculating embeddings. You should increase this if your GPU has tonnes of RAM!",
            default=25,
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
            default="reference_data/CN2025_SelfText_EN_DE_FR.csv",
        )
        parser.add_argument(
            "--brands-data-file",
            type=str,
            help="the path to the CN data file",
            default="reference_data/brands.csv",
        )
        parser.add_argument(
            "--incorrect-description-pairs-file",
            type=str,
            help="the path to the incorrect description pairs file",
            default="reference_data/incorrect_code_desc_pairs.csv",
        )
        parser.add_argument(
            "--phrases-to-remove-file",
            type=str,
            help="the path to the phrases to remove file",
            default="reference_data/phrases_to_remove.txt",
        )
        parser.add_argument(
            "--tradesets-data-dir",
            type=str,
            help="the path to the tradesets data directory",
            default="raw_source_data/tradesets_descriptions",
        )
        parser.add_argument(
            "--transformer",
            type=str,
            help="the transformer to use for generating the embeddings",
            default="all-mpnet-base-v2",
        )
        parser.add_argument(
            "--exact-english-terms",
            type=str,
            help="the path to the known english terms that match descriptions exactly.",
            default="reference_data/exact_english_terms.txt",
        )
        parser.add_argument(
            "--partial-english-terms",
            type=str,
            help="the path to the known english terms file. We preserve descriptions that match these words.",
            default="reference_data/partial_english_terms.txt",
        )
        parser.add_argument(
            "--partial-non-english-terms",
            type=str,
            help="the path to the known non-english terms file. We discard descriptions that match these words.",
            default="reference_data/partial_non_english_terms.txt",
        )
        parser.add_argument(
            "--preferred-languages",
            type=str,
            nargs="+",
            help="the preferred languages to keep",
            default=["ENGLISH"],
            required=False,
        )
        parser.add_argument(
            "--detected-languages",
            type=str,
            nargs="+",
            help="the detected languages to keep",
            default=["ENGLISH", "FRENCH", "GERMAN", "SPANISH"],
            required=False,
        )
        parser.add_argument(
            "--minimum-relative-distance",
            type=float,
            help="the minimum relative distance to use for language detection",
            default=0.7,
        )
        parser.add_argument(
            "--model-dropout-layer-1-percentage",
            type=float,
            help="the percentage of dropout to use in the first dropout layer",
            default=0.1,
        )
        parser.add_argument(
            "--model-dropout-layer-2-percentage",
            type=float,
            help="the percentage of dropout to use in the second dropout layer",
            default=0.3,
        )

        self.parsed_args, _unknown = parser.parse_known_args()
        self._parse_search_config()

    def print(self):
        logger.info("Configuration:")
        logger.info(f"  device: {self.device()}")
        logger.info(f"  torch_device: {self.torch_device()}")
        logger.info(f"  torch_version: {torch.__version__}")
        logger.info(f"  learning_rate: {self.learning_rate()}")
        logger.info(f"  max_epochs: {self.max_epochs()}")
        logger.info(f"  model_batch_size: {self.model_batch_size()}")
        logger.info(f"  embedding_batch_size: {self.embedding_batch_size()}")
        logger.info(f"  vague_terms_data_file: {self.vague_terms_data_file()}")
        logger.info(f"  limit: {self.limit()}")
        logger.info(f"  digits: {self.digits()}")
        logger.info(
            f"  extra_references_data_file: {self.extra_references_data_file()}"
        )
        logger.info(f"  cn_data_file: {self.cn_data_file()}")
        logger.info(f"  brands_data_file: {self.brands_data_file()}")
        logger.info(
            f"  incorrect_description_pairs_file: {self.incorrect_description_pairs_file()}"
        )
        logger.info(f"  tradesets_data_dir: {self.tradesets_data_dir()}")
        logger.info(f"  data_dir: {self.data_dir()}")
        logger.info(f"  target_dir: {self.target_dir()}")
        logger.info(f"  transformer: {self.transformer()}")
        logger.info(f"  partial_english_terms: {self.partial_english_terms()}")
        logger.info(f"  partial_non_english_terms: {self.partial_non_english_terms()}")
        logger.info(f"  exact_english_terms: {self.exact_english_terms()}")
        logger.info(f"  preferred_languages: {self.preferred_languages()}")
        logger.info(f"  detected_languages: {self.detected_languages()}")
        logger.info(f"  minimum_relative_distance: {self.minimum_relative_distance()}")

    def torch_device(self):
        arg_device = self.device()

        if arg_device == "cuda":
            return self._cuda_device()

        if arg_device == "mps":
            return self._mps_device()

        if arg_device == "auto":
            return self._auto_device()

        return arg_device

    def target_dir(self):
        return self.pwd() / "target"

    def data_dir(self):
        return self.target_dir() / "training_data"

    def reference_dir(self):
        return self.pwd() / "reference_data"

    def pwd(self):
        return Path(__file__).resolve().parent

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
    def brands_data_file(self):
        return self.parsed_args.brands_data_file

    @config_from_file
    def incorrect_description_pairs_file(self):
        return self.parsed_args.incorrect_description_pairs_file

    @config_from_file
    def phrases_to_remove_file(self):
        return self.parsed_args.phrases_to_remove_file

    @config_from_file
    def tradesets_data_dir(self):
        return self.parsed_args.tradesets_data_dir

    @config_from_file
    def transformer(self):
        return self.parsed_args.transformer

    @config_from_file
    def model_input_size(self):
        raise NotImplementedError(
            "Have you got an up-to-date model.pt and search-config.toml? search-config.toml includes model inputs after a model is generated."
        )

    @config_from_file
    def model_hidden_size(self):
        raise NotImplementedError(
            "Have you got an up-to-date model.pt and search-config.toml? search-config.toml includes model inputs after a model is generated."
        )

    @config_from_file
    def model_output_size(self):
        raise NotImplementedError(
            "Have you got an up-to-date model.pt and search-config.toml? search-config.toml includes model inputs after a model is generated."
        )

    @config_from_file
    def partial_english_terms(self):
        return self.parsed_args.partial_english_terms

    @config_from_file
    def partial_non_english_terms(self):
        return self.parsed_args.partial_non_english_terms

    @config_from_file
    def exact_english_terms(self):
        return self.parsed_args.exact_english_terms

    @config_from_file
    def preferred_languages(self):
        return self.parsed_args.preferred_languages

    @config_from_file
    def detected_languages(self):
        return self.parsed_args.detected_languages

    @config_from_file
    def minimum_relative_distance(self):
        return self.parsed_args.minimum_relative_distance

    @config_from_file
    def model_dropout_layer_1_percentage(self):
        return self.parsed_args.model_dropout_layer_1_percentage

    @config_from_file
    def model_dropout_layer_2_percentage(self):
        return self.parsed_args.model_dropout_layer_2_percentage

    def load_config_file(self):
        self.parsed_config = toml.load("search-config.toml")

    def _parse_search_config(self):
        if self.parsed_args.config is not None:
            self.parsed_config = toml.load(self.parsed_args.config)
        else:
            self.parsed_config = None

    def _auto_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _cuda_device(self):
        if not torch.cuda.is_available():
            return "cpu"

        return "cuda"

    def _mps_device(self):
        if not torch.backends.mps.is_available():
            return "cpu"

        return "mps"
