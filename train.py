if __name__ == "__main__":
    import logging
    import pickle
    from pathlib import Path

    import toml
    import torch

    from data_sources.basic_csv import BasicCSVDataSource
    from data_sources.commodities import CommoditiesDataSource
    from data_sources.data_source import DataSource
    from data_sources.search_references import SearchReferencesDataSource
    from data_sources.vague_terms import VagueTermsCSVDataSource
    from train_args import TrainScriptArgsParser
    from training.cleaners.map_2024_to_2025_codes import Map2024CodesTo2025Codes
    from training.cleaning_pipeline import (
        CleaningPipeline,
        DescriptionLower,
        IncorrectPairsRemover,
        LanguageCleaning,
        NegationCleaning,
        PhraseRemover,
        PluralCleaning,
        RemoveDescriptionsMatchingRegexes,
        RemoveEmptyDescription,
        RemoveShortDescription,
        RemoveSubheadingsNotMatchingRegexes,
        StripExcessCharacters,
    )
    from training.create_embeddings import EmbeddingsProcessor
    from training.prepare_data import TrainingDataLoader
    from training.train_model import FlatClassifierModelTrainer

    args = TrainScriptArgsParser()
    args.print()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train")

    cwd = Path(__file__).resolve().parent

    target_dir = args.target_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    args.reference_dir().mkdir(parents=True, exist_ok=True)

    # First load in the training data
    print("💾⇨ Loading training data")

    subheadings_file = target_dir / "subheadings.pkl"

    language_skips_file = args.pwd() / args.partial_non_english_terms()
    language_keeps_file = args.pwd() / args.partial_english_terms()
    language_keeps_exact_file = args.pwd() / args.exact_english_terms()

    with open(language_skips_file, "r") as f:
        language_skips = f.read().splitlines()

    with open(language_keeps_file, "r") as f:
        language_keeps = f.read().splitlines()

    with open(language_keeps_exact_file, "r") as f:
        language_keeps_exact = f.read().splitlines()

    basic_filters = [
        DescriptionLower(),
        StripExcessCharacters(),
        RemoveEmptyDescription(),
        RemoveShortDescription(min_length=1),
        RemoveSubheadingsNotMatchingRegexes(
            regexes=[
                "^\\d{" + str(args.digits()) + "}$",
            ]
        ),
        Map2024CodesTo2025Codes(),
    ]
    tradestats_filters = [
        DescriptionLower(),
        PhraseRemover.build(args.phrases_to_remove_file()),
        StripExcessCharacters(),
        RemoveEmptyDescription(),
        RemoveShortDescription(min_length=1),
        RemoveSubheadingsNotMatchingRegexes(
            regexes=[
                "^\\d{" + str(args.digits()) + "}$",
            ]
        ),
        RemoveDescriptionsMatchingRegexes.build(),
        LanguageCleaning(
            detected_languages=args.detected_languages(),
            preferred_languages=args.preferred_languages(),
            partial_skips=language_skips,
            partial_keeps=language_keeps,
            exact_keeps=language_keeps_exact,
        ),
        IncorrectPairsRemover.build(args.incorrect_description_pairs_file()),
        PluralCleaning(),
    ]

    self_texts_filters = basic_filters + [NegationCleaning.build()]

    basic_pipeline = CleaningPipeline(basic_filters)
    tradestats_pipeline = CleaningPipeline(tradestats_filters)
    self_texts_pipeline = CleaningPipeline(self_texts_filters)

    data_sources: list[DataSource] = []

    data_sources.append(
        VagueTermsCSVDataSource(
            args.vague_terms_data_file(),
            args.vague_terms_regex_file(),
        )
    )

    data_sources.append(
        BasicCSVDataSource(
            args.extra_references_data_file(),
            code_col=1,
            description_col=0,
            authoritative=True,
            creates_codes=False,
            multiplier=5,
        )
    )

    data_sources.append(
        BasicCSVDataSource(
            args.cn_data_file(),
            code_col=1,
            description_col=2,
            cleaning_pipeline=self_texts_pipeline,
            authoritative=True,
            creates_codes=True,
            multiplier=3,
        )
    )
    data_sources.append(SearchReferencesDataSource(multiplier=10))
    data_sources.append(CommoditiesDataSource(cleaning_pipeline=self_texts_pipeline))
    data_sources.append(
        BasicCSVDataSource(
            args.brands_data_file(),
            code_col=0,
            description_col=1,
            cleaning_pipeline=basic_pipeline,
            multiplier=3,
            authoritative=True,
            creates_codes=False,
        )
    )

    data_sources += [
        BasicCSVDataSource(
            filename,
            cleaning_pipeline=tradestats_pipeline,
            encoding="latin_1",
        )
        for filename in Path(args.tradesets_data_dir()).glob("*.csv")
    ]

    training_data_loader = TrainingDataLoader()

    (
        unique_text_values,
        subheadings,
        text_indexes,
        labels,
    ) = training_data_loader.fetch_data(data_sources, args.digits())

    print(f"Found {len(unique_text_values)} unique descriptions")

    print("💾⇦ Saving subheadings")
    with open(subheadings_file, "wb") as fp:
        pickle.dump(subheadings, fp)

    # Impose the limit if required - this will limit the number of unique descriptions
    if args.limit() is not None:
        unique_text_values = unique_text_values[: args.limit()]

        new_texts: list[int] = []
        new_labels: list[int] = []

        for i, t in enumerate(text_indexes):
            if t < len(unique_text_values):
                new_texts.append(t)
                new_labels.append(labels[i])

        text_indexes = new_texts
        labels = new_labels

    # Next create the embeddings
    print("Creating the embeddings")

    embeddings_processor = EmbeddingsProcessor(
        transformer_model=args.transformer(),
        torch_device=args.torch_device(),
        batch_size=args.embedding_batch_size(),
    )

    unique_embeddings = embeddings_processor.create_embeddings(unique_text_values)

    # Now build and train the network
    trainer = FlatClassifierModelTrainer(args)

    # Convert the labels to a Tensor
    labels = torch.tensor(labels, dtype=torch.long)

    embeddings = torch.stack([unique_embeddings[idx] for idx in text_indexes])

    state_dict, input_size, hidden_size, output_size = trainer.run(
        embeddings, labels, len(subheadings)
    )

    print("💾⇦ Saving model")

    model_file = target_dir / "model.pt"
    torch.save(state_dict, model_file)

    model_config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "dropout_layer_1_percentage": args.model_dropout_layer_1_percentage(),
        "dropout_layer_2_percentage": args.model_dropout_layer_2_percentage(),
    }

    with open("target/model.toml", "w") as f:
        toml.dump(model_config, f)

    print("✅ Training complete. Enjoy your model!")
