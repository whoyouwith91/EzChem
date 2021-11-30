import torch
import torch.nn as nn

from torch_scatter import scatter


class EdgeToAtomLayer(nn.Module):
    def __init__(self, source_to_target=True):
        """
        :param source_to_target: If true, assume edge_index[0, :] is the source and edge_index[1, :] is target
        """
        super().__init__()
        self.source_to_target = source_to_target

    def forward(self, edge_attr, edge_index):
        if self.source_to_target:
            return scatter(reduce='add', src=edge_attr, index=edge_index[1, :], dim=-2)
        else:
            return scatter(reduce='add', src=edge_attr, index=edge_index[0, :], dim=-2)


if __name__ == '__main__':
    pass
