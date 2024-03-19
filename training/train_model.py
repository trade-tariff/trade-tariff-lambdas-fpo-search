import math
import torch
from torch import Tensor, optim, nn
from torch.utils.data import DataLoader, TensorDataset
from model.model import SimpleNN
from sklearn.metrics import classification_report
import csv

class ModelTrainer:
    def run(self, X_train: Tensor, X_test: Tensor, y_train: Tensor, y_test: Tensor, num_labels: int) -> nn.Module:
        raise NotImplementedError()


class FlatClassifierModelTrainerParameters:
    def __init__(self, learning_rate: float = 0.001, max_epochs: int = 4) -> None: #edit max_epochs here
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

    def run(self, X_train: Tensor, X_test: Tensor, y_train: Tensor, y_test: Tensor, num_labels: int) -> nn.Module: 
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        input_size = len(X_train[0])  # Assuming embeddings have fixed size
        output_size = num_labels  # Number of unique classes in your labels
        hidden_size = math.floor(
            input_size + (output_size - input_size) / 2
        )  # You can adjust this as needed
        model = SimpleNN(input_size, hidden_size, output_size).to(self._device)

        # Create data loaders.
        batch_size = 1000  # Adjust as needed
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        loss_fn = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(model.parameters(), lr=self._parameters.learning_rate)
        
        print("Created model")
        print(model)

        max_epochs = self._parameters.max_epochs

        def train(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset) #get the length of dataloader’s dataset
            model.train()
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self._device), y.to(self._device)
                
                # Compute prediction error
                pred = model(X) 
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() #makes sure to empty the gradient after/ in each iteration

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        def test(dataloader, model, loss_fn):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            model.eval()
            test_loss, correct = 0, 0

            #For classification report
            all_labels = []
            all_predictions = []

            with torch.no_grad(): #stops tensor from tracking history so it's not part of the gradient computation
                for X, y in dataloader:
                    X, y = X.to(self._device), y.to(self._device)
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    _, predicted = torch.max(pred, 1)
                    all_labels.extend(y.cpu().numpy()) #move tensor to cpu so can convert to numpy
                    all_predictions.extend(predicted.cpu().numpy()) #move tensor to cpu so can convert to numpy


            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            #Calculate the classification report
            report = classification_report(all_labels, all_predictions)
            report_dict = classification_report(all_labels, all_predictions, output_dict=True)
            return report_dict

        def save_accuracy_to_csv(report_dict, file_path='classification_results.csv'):
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = ['class_label', 'precision', 'recall', 'f1score', 'support']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                # Debugging: Print the metrics for each class
                #print(f"Class {class_label} Metrics:", metrics)
                for class_label, metrics in report_dict.items():
                    # Check if metrics is a dictionary
                    if isinstance(metrics, dict):
                        writer.writerow({
                            'class_label': class_label,
                            'precision': metrics.get('precision', None),
                            'recall': metrics.get('recall', None),
                            'f1score': metrics.get('f1-score', None),
                            'support': metrics.get('support', None)
                        })
                    else:
                        # Handle the case where metrics is a float
                        writer.writerow({'class_label': class_label, 'precision': metrics, 'recall': None, 'f1score': None, 'support': None
                        })

        for t in range(max_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer)
            report_dict= test(test_loader, model, loss_fn)
            save_accuracy_to_csv(report_dict)
        print("Done!")

        return model
