import math
import torch
from torch import Tensor, optim, nn
from torch.utils.data import DataLoader, TensorDataset
from model.model import SimpleNN


class ModelTrainer:
    def run(self, embeddings: Tensor, labels: Tensor, num_labels: int) -> nn.Module:
        raise NotImplementedError()


class FlatClassifierModelTrainerParameters:
    def __init__(self, learning_rate: float = 0.001, max_epochs: int = 3) -> None:
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs


class FlatClassifierModelTrainer(ModelTrainer):
    def __init__(
        self,
        parameters: FlatClassifierModelTrainerParameters = FlatClassifierModelTrainerParameters(),
        device: str = "cpu",
    ) -> None:
        self._parameters = parameters
        self._device = device

    def run(self, embeddings: Tensor, labels: Tensor, num_labels: int) -> nn.Module:
        train_dataset = TensorDataset(embeddings, labels)

        input_size = len(embeddings[0])  # Assuming embeddings have fixed size
        output_size = num_labels  # Number of unique classes in your labels
        hidden_size = math.floor(
            input_size + (output_size - input_size) / 2
        )  # You can adjust this as needed
        model = SimpleNN(input_size, hidden_size, output_size).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._parameters.learning_rate)

        print("Created model")
        print(model)

        batch_size = 1000  # Adjust as needed
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        batches = len(train_loader)
        max_epochs = self._parameters.max_epochs

        for epoch in range(max_epochs):
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

                print(
                    f"Epoch {epoch+1}/{max_epochs}, Batch {i + 1}/{batches}, Acc: {100 * correct / total}%"
                )

        return model
