import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out)
        out = self.fc(out)

        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        self.dropout = nn.Dropout(0.1)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    # 示例使用
    batch_size = 10
    sequence_length = 60
    num_features = 7
    hidden_size = 64
    num_layers = 2
    output_size = 1

    # 创建输入数据
    input_data = torch.randn(batch_size, sequence_length, num_features)

    # 创建 LSTM 模型
    model = LSTMModel(num_features, hidden_size, num_layers, output_size)
    model2 = GRUModel(num_features, hidden_size, num_layers, output_size)

    # 调用模型
    output = model(input_data)
    output2 = model2(input_data)

    print(output.shape)
    print(output2.shape)
