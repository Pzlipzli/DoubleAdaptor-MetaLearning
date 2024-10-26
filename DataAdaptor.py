import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataTransformer(nn.Module):
    def __init__(self, input_size, num_trans):
        super(DataTransformer, self).__init__()
        self.input_size = input_size
        self.y_size = 1
        self.num_trans = num_trans

        # Define a linear transformation for each transformation
        self.linear_transforms_x = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_trans)])
        self.linear_transforms_y = nn.ModuleList([nn.Linear(self.y_size, self.y_size) for _ in range(num_trans)])

        # Define random_vectors as a learnable parameter without batch dimension
        self.p_vectors = nn.Parameter(torch.rand(num_trans, 1, input_size))
        self.temp_similarities = None

    def forward(self, input_data, is_y: bool = False):
        if not is_y:
            if self.temp_similarities is not None:
                warnings.warn("An x has been put into the transformer, but the similarities are not used yet."
                              "Check if the related y should have been used.")

            # Apply linear transformation
            transformed_data = torch.stack([self.linear_transforms_x[i](input_data) for i in range(self.num_trans)],
                                           dim=0)
            # Compute cosine similarities along the time dimension
            input_norm = torch.norm(input_data, dim=2, keepdim=True).unsqueeze(0)
            p_vectors_norm = torch.norm(self.p_vectors, dim=2, keepdim=True).unsqueeze(1)

            dot_product = torch.sum(input_data.unsqueeze(0) * self.p_vectors.unsqueeze(1), dim=3, keepdim=True)
            cosine_similarities = dot_product / (input_norm * p_vectors_norm)

            # Apply softmax to the cosine similarities
            softmax_similarities = F.softmax(cosine_similarities, dim=0)
        else:
            transformed_data = torch.stack([self.linear_transforms_y[i](input_data) for i in range(self.num_trans)],
                                           dim=0)
            if self.temp_similarities is not None:
                softmax_similarities = self.temp_similarities
            else:
                raise "x must be put into the transformer before y"

        # Compute the weighted sum
        weighted_sum = torch.sum(softmax_similarities * transformed_data, dim=0, keepdim=False)

        if not is_y:
            weighted_sum += input_data
            self.temp_similarities = softmax_similarities
        else:
            self.temp_similarities = None

        return weighted_sum

    def inverse_transform(self, transformed_data):
        # Compute the inverse of the linear transformation
        input_data = transformed_data.clone()

        inverse_transforms = torch.stack(
            [torch.linalg.pinv(self.linear_transforms_y[i].weight) for i in range(self.num_trans)], dim=0)
        inverse_biases = torch.stack([self.linear_transforms_y[i].bias for i in range(self.num_trans)], dim=0)

        # Apply inverse transformation
        input_data = torch.stack([torch.matmul((input_data - inverse_biases[i]), inverse_transforms[i].T)
                                        for i in range(self.num_trans)], dim=0)

        if self.temp_similarities is not None:
            softmax_similarities = self.temp_similarities
        else:
            raise "x must be put into the transformer before y"

        # Compute the weighted sum
        inverse_sum = torch.sum(softmax_similarities * input_data, dim=0, keepdim=False)
        self.temp_similarities = None

        return inverse_sum


if __name__ == "__main__":
    # 示例使用
    batch_size = 30
    sequence_length = 20
    num_features = 360
    num_trans = 8

    # 创建输入数据
    input_data = torch.randn(batch_size, sequence_length, num_features)

    # 创建数据转换器
    data_transformer = DataTransformer(num_features, num_trans)

    # 调用转换函数
    weighted_sum = data_transformer(input_data)
    recovered_data = data_transformer.inverse_transform(weighted_sum[:, :, 0:1])

    print("Weighted Sum Shape:", weighted_sum.shape)  # 输出应该是 (batch_size, sequence_length, num_features)
    # print("Recovered Data Shape:", recovered_data.shape)  # 输出应该是 (batch_size, sequence_length, num_features)
    # for name, param in data_transformer.named_parameters():
    #     print(f"Parameter name: {name}, requires_grad: {param.requires_grad}, shape: {param.shape}")
