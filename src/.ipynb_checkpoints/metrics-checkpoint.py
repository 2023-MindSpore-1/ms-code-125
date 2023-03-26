from mindspore import nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.parameter import ParameterTuple
from mindspore import ops
import mindspore as ms
import numpy as np

class Loss_GraphAUC(nn.Cell):
    def __init__(self, num_classes, num_nodes, adj_maxtrix, global_effect_matrix, global_perclass_mean_effect_matrix, mask, param, weight_sub_dim=64, weight_inter_dim=64, weight_global_dim=64, beta=0.5, gamma=1, is_ner_weight=True, loss_type="ExpGAUC"):
        super(Loss_GraphAUC, self).__init__(auto_prefix=False)
        
        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta
        self.loss_type = loss_type
        self.is_ner_weight = is_ner_weight
        self.mask = mask
        self.param = param

        self.num_nodes = num_nodes
        self.weight_sub_dim = weight_sub_dim
        self.weight_inter_dim = weight_inter_dim
        self.weight_global_dim = weight_global_dim
        self.adj_maxtrix = adj_maxtrix
        self.global_effect_matrix = self.int_index(Tensor(global_effect_matrix), Tensor(mask))
        self.global_perclass_mean_effect_matrix=global_perclass_mean_effect_matrix #[N,C]

        self.eye = ops.Eye()
        self.I = self.eye(self.num_nodes, self.num_nodes, ms.bool_)#.to(self.device)
        inn = Tensor(adj_maxtrix, dtype=ms.int32)
        inn = inn.asnumpy()
        nei = np.diagonal(inn)
        jie = np.diag(nei)
        nobo = (adj_maxtrix.asnumpy())^jie
        tem = Tensor(nobo, ms.bool_)
        tem = np.array(nobo, dtype = bool)
        I_np= self.I.asnumpy()
        adj_self_matrix = tem|I_np
        self.adj_self_matrix = Tensor(adj_self_matrix)

        self.stack = ops.Stack()
        self.transpose = ops.Transpose()
        self.squeeze = ops.Squeeze()
        self.expand_dims = ops.ExpandDims()
        self.exp = ops.Exp()
        self.sequal = ops.Equal()
        self.concat_op = ops.Concat()
        self.cast_op = ops.Cast()
        self.reshape = ops.Reshape()
        self.sigmoid = ops.Sigmoid()
        self.l2_loss = ops.L2Loss()
        self.reduce_sum = ops.ReduceSum(False)
        self.mean = ops.ReduceMean()

    def int_index(self, nums, indexs):
        empty_list = []
        nums_np = nums.asnumpy()
        indexs_np=indexs.asnumpy()
        
        for i in range(nums_np.shape[0]):
            line = nums_np[i]
            empty_line = []
            for j in range(indexs_np.shape[0]):
                if indexs_np[j] == 1:
                    empty_line.append(nums_np[i][j])
            empty_list.append(empty_line)
        res = Tensor(empty_list)
        return res
    
    def get_pred(self, preds, mask):
        empty_list = []
        preds_np = preds.asnumpy()
        mask_np = mask.asnumpy()
        for i in range(mask_np.shape[0]):
            if mask_np[i] == 1:
                empty_list.append(preds_np[i])
        res = Tensor(empty_list)
        return res
    
    def get_tem_label(self, label, mask):
        label_np = label.asnumpy()
        mask_np  = mask.asnumpy()
        empty_list = []
        for i in range(mask_np.shape[0]):
            if mask_np[i]==1:
                empty_list.append(label_np[i])
        res = Tensor(empty_list)
        return res
    
    def get_label(self, tem_label):
        empty_list = []
        tem_label_np = tem_label.asnumpy()
        for line in tem_label:
            for i in range(line.shape[0]):
                if line[i]==1:
                    empty_list.append(i)
        res = Tensor(empty_list)
        return res
    
    def show(self, item):
        print(item, type(item))
        print(item.shape)
        
    def nonzero(self, inp):
        inp_np = inp.asnumpy()
        tem = np.transpose(np.nonzero(inp_np))
        res = Tensor(tem)
        return res
    
    def nonzero_tuple(self, inp):
        inp_np = inp.asnumpy()
        b_np = np.nonzero(inp_np)
        empty_list = []
        for i in b_np:
            empty_list.append(Tensor(i))
        return tuple(empty_list)

    def construct(self, preds, label):
        pred = self.get_pred(Tensor(preds), Tensor(self.mask))
        tem_label = self.get_pred(Tensor(label), Tensor(self.mask))
        label = self.get_label(Tensor(tem_label))
        
        label_np = label.asnumpy()
        tem = np.stack(
            [np.equal(label_np,i) for i in range(self.num_classes)],1
        )
        Y_np = np.squeeze(tem).astype(float)
        N_np = Y_np.sum(0)
        Y = Tensor(Y_np)
        N = Tensor(N_np) 
        loss = Tensor([0.])
        
        #self.global_sub=self.linear_sub(self.global_effect_matrix).sum(axis=-1)
        #self.global_inter=self.linear_inter(self.global_effect_matrix).sum(axis=-1)
        #self.global_global=self.linear_global(self.global_effect_matrix).sum(axis=-1)
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i!=j:
                    i_pred_pos = Tensor(pred.asnumpy()[Y[:,i].asnumpy().astype(bool), :][:, i])
                    i_pred_neg = Tensor(pred.asnumpy()[Y[:,j].asnumpy().astype(bool), :][:, i])
                    broadcast_to_0 = ops.BroadcastTo((i_pred_pos.shape[0],i_pred_neg.shape[0]))
                    i_pred_pos_expand = broadcast_to_0(self.expand_dims(i_pred_pos,1))
                    i_pred_pos_sub_neg = i_pred_pos_expand-i_pred_neg
                    ij_loss = self.exp(-self.gamma * i_pred_pos_sub_neg)
                    
                    i_pred_pos_index = self.nonzero(Tensor(Y[:,i])).view(-1)
                    i_pred_neg_index = self.nonzero(Tensor(Y[:,j])).view(-1)
                    
                    adj_maxtrix_np = self.adj_maxtrix.asnumpy()
                    adj_self_matrix_np = self.adj_self_matrix.asnumpy()
                    mask_np = self.mask.asnumpy()
                    i_pred_pos_index_np = i_pred_pos_index.asnumpy()
                    i_pred_neg_index_np = i_pred_neg_index.asnumpy()

                    i_pred_pos_adj = Tensor(adj_maxtrix_np[mask_np][i_pred_pos_index_np])
                    i_pred_neg_adj = Tensor(adj_maxtrix_np[mask_np][i_pred_neg_index_np])
                    i_pred_neg_self_adj = Tensor(adj_self_matrix_np[mask_np][i_pred_neg_index_np])
                
                    broadcast_to_1 = ops.BroadcastTo((i_pred_pos_adj.shape[0],i_pred_neg_adj.shape[0],i_pred_pos_adj.shape[1]))
                    i_pred_pos_adj_expand = broadcast_to_1(self.expand_dims(i_pred_pos_adj,1))
                    sub_ner=Tensor((i_pred_pos_adj_expand.asnumpy()^(i_pred_pos_adj_expand.asnumpy()&i_pred_neg_self_adj.asnumpy())))
                    inter_ner=Tensor((i_pred_pos_adj_expand.asnumpy()&i_pred_neg_adj.asnumpy()))
                    if (np.count_nonzero(inter_ner.asnumpy())>0) and ((np.count_nonzero(sub_ner.asnumpy())>0)):
                        sub_ner_nonzero = self.nonzero_tuple(Tensor(sub_ner))    
                        a = self.cast_op(sub_ner_nonzero[0],ms.int32)
                        b = self.cast_op(sub_ner_nonzero[1],ms.int32)
                        I_sub_tem = self.concat_op((a,b))
                        I_sub = self.reshape(I_sub_tem, (2,-1))
                        I_sub = I_sub.T
                        V_sub = self.sigmoid( self.cast_op(sub_ner_nonzero[2], ms.float32))
                        S_sub = ms.COOTensor(I_sub, V_sub, sub_ner.shape[:-1])
                        vi_sub = S_sub.to_dense()

                        inter_ner_nonzero = self.nonzero_tuple(inter_ner)
                        a = self.cast_op(inter_ner_nonzero[0],ms.int32) 
                        b = self.cast_op(inter_ner_nonzero[1],ms.int32)
                        I_inter_tem = self.concat_op((a,b))
                        I_inter = self.reshape(I_inter_tem,(2,-1))
                        I_inter = I_inter.T
                        V_inter = self.sigmoid( self.cast_op(inter_ner_nonzero[2], ms.float32))
                        S_inter = ms.COOTensor(I_inter, V_inter, inter_ner.shape[:-1])
                        vi_inter = S_inter.to_dense()
                    
                        vl_i = self.sigmoid((1+vi_sub)/(1+vi_inter))
                        v_i = 1-vl_i
                        ij_loss = self.reduce_sum( (1/(N[i]*N[j])*v_i*ij_loss) )
                        loss+=ij_loss 
        return loss
        