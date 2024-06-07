import logging
import torch
from torch import Tensor, optim, nn
from torch.utils.data import DataLoader, TensorDataset
from model.model import SimpleNN
from typing import Any, Dict
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
        self. dropout_prob1 = args.model_dropout_layer_1_percentage()
        self. dropout_prob2 = args.model_dropout_layer_2_percentage()

    def run(
        self, embeddings: Tensor, labels: Tensor, num_labels: int
    ) -> tuple[Dict[str, Any], int, int, int]:
        train_dataset = TensorDataset(embeddings, labels)

        input_size = len(embeddings[0])  # Assuming embeddings have fixed size
        output_size = num_labels  # Number of unique classes in your labels
        hidden_size = int(0.8 * (input_size + output_size))

        model = SimpleNN(
            input_size, hidden_size, output_size, self.dropout_prob1, self.dropout_prob2
        ).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._learning_rate)

        logger.info("Created model")
        logger.info(model)

        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )

        batches = len(train_loader)

        for epoch in range(self._max_epochs):
            model.train()
            running_loss = 0.0
            total = 0
            correct = 0

            for i, (inputs, loader_labels) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs = inputs.to(self._device)
                loader_labels = loader_labels.to(self._device)

                outputs = model(inputs)

                loss = criterion(outputs, loader_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == loader_labels).sum().item()

                # Explicitly remove the tensors from the GPU
                del inputs
                del loader_labels

                logger.info(
                    f"Epoch {epoch+1}/{self._max_epochs}, Batch {i + 1}/{batches}, Acc: {100 * correct / total}%"
                )

        return (
            model.to("cpu").state_dict(),
            input_size,
            hidden_size,
            output_size,
        )
