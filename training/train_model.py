import math
import torch
from torch import Tensor, optim, nn
from torch.utils.data import DataLoader, TensorDataset
from training.model import SimpleNN


class ModelTrainer:
    def __init__(
        self, learning_rate: float = 0.001, max_epochs: int = 3, device: str = "cpu"
    ) -> None:
        self._learning_rate = learning_rate
        self._max_epochs = max_epochs
        self._device = device

    def run(self, embeddings: Tensor, labels: Tensor, num_labels: int) -> nn.Module:
        train_dataset = TensorDataset(embeddings, labels)

        input_size = len(embeddings[0])  # Assuming embeddings have fixed size
        output_size = num_labels  # Number of unique classes in your labels
        hidden_size = math.floor(
            input_size + (output_size - input_size) / 2
        )  # You can adjust this as needed
        model = SimpleNN(input_size, hidden_size, output_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._learning_rate)

        print("Created model")
        print(model)

        batch_size = 1000  # Adjust as needed
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        batches = len(train_loader)

        model.to(self._device)

        for epoch in range(self._max_epochs):
            model.train()
            running_loss = 0.0
            total = 0
            correct = 0

            for i, (inputs, loader_labels) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs.to(self._device)
                loader_labels.to(self._device)

                outputs = model(inputs)

                loss = criterion(outputs, loader_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == loader_labels).sum().item()

                print(
                    f"Epoch {epoch+1}/{self._max_epochs}, Batch {i + 1}/{batches}, Acc: {100 * correct / total}%"
                )

        return model
