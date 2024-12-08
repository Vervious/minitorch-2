"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# Implement for Task 2.5.

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        """hidden_layers is size of hidden layer, not number of layers (2)."""
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        middle = self.layer1.forward(x).relu()
        end = self.layer2.forward(middle).relu()
        return self.layer3.forward(end).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.weights = RParam(in_size, out_size)
        self.weights.name = "weights"

        self.bias = RParam(out_size,)
        self.bias.name = "bias"


    def forward(self, inputs):
        # should really do matrix mults here...
        # print("inputs", inputs.shape)
        # print("input mod shape", inputs.view(*inputs.shape, 1).shape)
        # print("weights shape", self.weights.shape)
        matmul = inputs.view(*inputs.shape, 1) * self.weights.value
        # print("eement mult", matmul.shape)
        matmul = matmul.sum(1)
        # print("post sum", matmul.shape)
        matmul = matmul.view(*matmul.shape[:-2], matmul.shape[-1])
        # print("post dim reduction", matmul.shape)

        y = self.bias.value + matmul
        # print("bais shape", self.bias.shape)
        # print("y shape", y.shape)
        return y

        # batch, in_size = x.shape
        # matmul = x.view(batch, in_size, 1) * self.weights.value.view(1, in_size, self.out_size)
        # matmul = matmul.sum(1)
        # matmul = matmul.view(batch, self.out_size)
        # y = matmul + self.bias.value.view(self.out_size)
        # return y


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        # self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 10
    HIDDEN = 3
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    model = TensorTrain(HIDDEN)
    model.train(data, RATE, max_epochs=100)
