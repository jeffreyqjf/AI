import torch.nn as nn
import torch
"""
I am jeffrey. I want to learn more about RNN and other networks.
"""
dictionary = {
    "I": torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "am": torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "jeffrey": torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ".": torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "want": torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    "to": torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    "learn": torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    "more": torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    "about": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    "RNN": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    "and": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    "other": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    "networks": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
}
class MyRNN_cell(nn.Module):
    def __init__(self, in_features, out_features):
        self.linear_bef = nn.Linear(in_features, out_features)
        self.linear_ori = nn.Linear(in_features, out_features)
        self.tanh =  nn.Tanh()

    def forward(self, input, input_before):
        output_ori = self.linear_ori(input)
        output_bef = self.linear_bef(input_before)
        return self.tanh(output_bef + output_ori)


class MyRNN(nn.Module):
    def __init__(self, in_features, out_features, layers):
        self.layers = layers
        self.rnn = []
        for i in range(layers):
            self.rnn.append(MyRNN_cell(in_features, out_features))
        self.input_bef_0 = torch.zeros() #!!!!

    def forward(self, input):
        for i in range(self.layers):
            if i == 0:
                output = self.rnn[i](input, self.input_bef_0)
            else:
                output = self.rnn[i](input, output)


if __name__ == "__main__":
    print(torch.tensor([0, 0, 1, 0]))

