import numpy as np
from mindspore import nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.nn.layer.activation import get_activation

def glorot(shape):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = np.random.uniform(-init_range, init_range, shape).astype(np.float32)
    return Tensor(initial)

class GraphConvolution(nn.Cell):
    def __init__(self,
                 feature_in_dim,
                 feature_out_dim,
                 dropout_ratio=None,
                 activation=None):
        super(GraphConvolution, self).__init__()
        self.in_dim = feature_in_dim
        self.out_dim = feature_out_dim
        self.weight_init = glorot([self.out_dim, self.in_dim])
        self.fc = nn.Dense(self.in_dim,
                           self.out_dim,
                           weight_init=self.weight_init,
                           has_bias=False)
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio is not None:
            self.dropout = nn.Dropout(keep_prob=1-self.dropout_ratio)
        self.dropout_flag = self.dropout_ratio is not None
        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None
        self.matmul = P.MatMul()

    def construct(self, adj, input_feature):
        dropout = input_feature
        if self.dropout_flag:
            dropout = self.dropout(dropout)

        fc = self.fc(dropout)
        output_feature = self.matmul(adj, fc)

        if self.activation_flag:
            output_feature = self.activation(output_feature)
        return output_feature

class StandGCN1(nn.Cell):
    def __init__(self, config, input_dim, output_dim, adj):
        super(StandGCN3, self).__init__()
        self.layer1 = GraphConvolution(input_dim, output_dim, activation="softmax", dropout_ratio=None)
        self.adj = adj 
    def construct(self, feature):
        output0 = self.layer0(self.adj, feature)
        return output0

class StandGCN2(nn.Cell):
    def __init__(self, config, input_dim, output_dim, adj):
        super(StandGCN3, self).__init__()
        self.layer0 = GraphConvolution(input_dim, config.hidden1, activation="relu", dropout_ratio=config.dropout)
        self.layer1 = GraphConvolution(config.hidden1, output_dim, activation="softmax", dropout_ratio=None)
        self.adj = adj 
    def construct(self, feature):
        output0 = self.layer0(self.adj, feature)
        output1 = self.layer1(self.adj, output0)
        return output1

class StandGCN3(nn.Cell):
    def __init__(self, config, input_dim, output_dim, adj):
        super(StandGCN3, self).__init__()
        self.layer0 = GraphConvolution(input_dim, config.hidden1, activation="relu", dropout_ratio=config.dropout)
        self.layer1 = GraphConvolution(config.hidden1, config.hidden2, activation="relu", dropout_ratio=config.dropout)
        self.layer2 = GraphConvolution(config.hidden1, output_dim, activation="softmax", dropout_ratio=None)
        self.adj = adj 
    def construct(self, feature):
        output0 = self.layer0(self.adj, feature)
        output1 = self.layer1(self.adj, output0)
        output2 = self.layer2(self.adj, output1)
        return output2