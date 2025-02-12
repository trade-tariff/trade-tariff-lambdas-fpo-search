from torch import nn


class ClassifierModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_prob1: float,
        dropout_prob2: float,
    ):
        super(ClassifierModel, self).__init__()
        self.dropout1 = nn.Dropout(
            p=dropout_prob1
        )  # Add dropout layer on incoming (visible) units
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob2)  # Keep the dropout layer
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Apply He initialization to fc1 weights
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        # Apply He initialization to fc2 weights
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        x = self.dropout1(x)  # Apply dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)  # Apply dropout
        x = self.fc2(x)
        return x
