import json
import logging
import math
from typing import Any

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

from model.model import SimpleNN
from train_args import TrainScriptArgsParser

logger = logging.getLogger("train")


class FlatClassifierModelTrainer:
    def __init__(
        self,
        args: TrainScriptArgsParser,
    ) -> None:
        self._device = args.torch_device()
        self._max_epochs = args.max_epochs()
        self._learning_rate = args.learning_rate()
        self._batch_size = args.model_batch_size()
        self._dropout_prob1 = args.model_dropout_layer_1_percentage()
        self._dropout_prob2 = args.model_dropout_layer_2_percentage()

    def run(
        self,
        embeddings: Tensor,
        labels: Tensor,
        num_labels: int,
        text_indexes: Tensor | None = None,
    ) -> tuple[dict[str, Any], int, int, int]:
        """Train on one embedding per label.

        With text_indexes, embeddings holds one row per unique text and each
        label uses embeddings[text_indexes[i]]. This avoids a copy per row.
        """
        if text_indexes is None:
            train_dataset = TensorDataset(embeddings, labels)
        else:
            train_dataset = TensorDataset(text_indexes, labels)

        input_size = len(embeddings[0])  # Assuming embeddings have fixed size
        output_size = num_labels  # Number of unique classes in your labels
        hidden_size = int(0.8 * (input_size + output_size))

        model = SimpleNN(
            input_size,
            hidden_size,
            output_size,
            self._dropout_prob1,
            self._dropout_prob2,
        ).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._learning_rate)

        logger.info("Created model")
        logger.info(model)

        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )
        total_steps = len(train_loader) * self._max_epochs

        # Learning rate warm up
        def lr_lambda(current_step: int):
            warmup_steps = 0.05 * total_steps  # 5% of total steps for warmup
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (current_step - warmup_steps)
                    / float(max(1, total_steps - warmup_steps))
                )
            )

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        batches = len(train_loader)
        size = len(train_dataset)

        report: dict[str, dict[str, float]] = {
            f"epoch_{epoch + 1}": {"accuracy": 0.0, "average_loss": 0.0}
            for epoch in range(self._max_epochs)
        }

        for epoch in range(self._max_epochs):
            model.train()
            running_loss = 0.0
            total = 0
            correct = 0

            for _, (inputs, loader_labels) in enumerate(train_loader):
                optimizer.zero_grad()

                if text_indexes is not None:
                    inputs = embeddings[inputs.to(embeddings.device)]
                inputs = inputs.to(self._device, torch.float32)
                loader_labels = loader_labels.to(self._device)

                outputs = model(inputs)

                loss = criterion(outputs, loader_labels)
                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate (learning rate warm up)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == loader_labels).sum().item()

                # Explicitly remove the tensors from the GPU
                del inputs
                del loader_labels

            running_loss /= batches
            correct /= size
            report[f"epoch_{epoch + 1}"]["accuracy"] = 100 * correct
            report[f"epoch_{epoch + 1}"]["average_loss"] = running_loss
            logger.info(
                f"{epoch + 1} \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {running_loss:>8f} \n"
            )

        with open("running_losses.json", "w") as f:
            f.write(json.dumps(report, indent=2))

        return (model.to("cpu").state_dict(), input_size, hidden_size, output_size)
