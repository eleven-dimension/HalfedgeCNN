import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.half_edge.neighbor import *
from packages.network.layer import *


class HalfEdgeCNNModel(nn.Module):
    def __init__(self, in_channel_num, mid_channel_num, category_num, neighbor_type_list):
        super(HalfEdgeCNNModel, self).__init__()
        self.neighbor_type_list = neighbor_type_list

