# -*- coding: utf-8 -*-
# Copied and adapted from AllenNLP

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BilinearMatrixAttention(nn.Module):
    """
    Computes attention between two matrices using a bilinear attention function.  This function has
    a matrix of weights ``W`` and a bias ``b``, and the similarity between the two matrices ``X``
    and ``Y`` is computed as ``X W Y^T + b``.

    Parameters
    ----------
    matrix_1_dim : ``int``
        The dimension of the matrix ``X``, described above.  This is ``X.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_2_dim : ``int``
        The dimension of the matrix ``Y``, described above.  This is ``Y.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    use_input_biases : ``bool``, optional (default = False)
        If True, we add biases to the inputs such that the final computation
        is equivalent to the original bilinear matrix multiplication plus a
        projection of both inputs.
    """
    def __init__(self,
                 matrix_1_dim: int,
                 matrix_2_dim: int,
                 use_input_biases: bool = False) -> None:
        super().__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        self._weight_matrix = Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))

        self._use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:

        if self._use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], -1)
            matrix_2 = torch.cat([matrix_2, bias2], -1)

        weight = self._weight_matrix
        ret = torch.matmul(matrix_1, weight.unsqueeze(0))
        ret = torch.matmul(ret, matrix_2.transpose(1, 2))

        return ret
