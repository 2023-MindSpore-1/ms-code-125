from mindspore import nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import ops
import mindspore as ms
import numpy as np
import mindspore.numpy as np

class e_loss_FN(nn.Cell):
    def __init__(self, num_classes, num_nodes, adj_maxtrix, global_effect_matrix, global_perclass_mean_effect_matrix, mask, param, weight_sub_dim=64, weight_inter_dim=64, weight_global_dim=64, beta=0.5, gamma=1, is_ner_weight=True, loss_type="ExpGAUC", per=1e-3):
        super(e_loss_FN, self).__init__(auto_prefix=False)

        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta
        self.loss_type = loss_type
        self.is_ner_weight = is_ner_weight
        self.mask = mask
        self.param = param
        self.per = per

        self.num_nodes = num_nodes
        self.weight_sub_dim = weight_sub_dim
        self.weight_inter_dim = weight_inter_dim
        self.weight_global_dim = weight_global_dim
        self.adj_maxtrix = adj_maxtrix
        #self.global_effect_matrix = global_effect_matrix[:,self.mask]
        self.global_effect_matrix = self.gem_cut(Tensor(global_effect_matrix), Tensor(mask))
        self.global_perclass_mean_effect_matrix=global_perclass_mean_effect_matrix #[N,C]

        self.eye = ops.Eye()
        self.I = self.eye(self.num_nodes, self.num_nodes, ms.bool_)#.to(self.device)



        hou = np.diag(np.diagonal(np.array(Tensor(adj_maxtrix, dtype=ms.int32))))
        hou = Tensor(hou, ms.bool_)
        qian = np.logical_xor(np.array(adj_maxtrix), np.array(hou))
        adj_self_matrix = np.logical_or(np.array(qian), np.array(self.I))

        adj_self_matrix = Tensor(adj_self_matrix)
        self.adj_self_matrix = adj_self_matrix

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
        self.sceloss = P.SoftmaxCrossEntropyWithLogits()
        self.mean = ops.ReduceMean()
        self.l2_loss = P.L2Loss()

        print("GAUC,OK")

    def gem_cut(self, gem, mask):
        gem = np.transpose(np.array(gem))
        mask= np.array(mask)
        lines = []
        for i in range(mask.shape[0]):
            if mask[i] == True:
                lines.append(gem[i])

        return np.transpose(np.array(lines))

    def get_pred(self, preds, mask):
        empty_list = []

        preds_np = np.array(preds)
        mask_np = np.array(mask)

        for i in range(mask_np.shape[0]):
            if mask_np[i] == 1:
                empty_list.append(preds_np[i])
        res = Tensor(np.array(empty_list))
        return res
    
    def get_tem_label(self, label, mask):
        label_np = np.array(label)
        mask_np  = np.array(mask)
        empty_list = []
        for i in range(mask_np.shape[0]):
            if mask_np[i]==1:
                empty_list.append(label_np[i])
        res = Tensor(empty_list)
        return res
    
    def get_label(self, tem_label):
        empty_list = []
        tem_label_np = np.array(tem_label)
        for line in tem_label:
            for i in range(line.shape[0]):
                if line[i]==1:
                    empty_list.append(i)
        res = Tensor(empty_list)
        return res
    
    def show(self, item):
        print(item, type(item))
        print(item.shape)

    def nonzero_tuple(self, inp):
        inp_np = np.array(inp)
        b_np = ops.nonzero(inp_np)
        tb_np = np.transpose(b_np)
        empty_list = []
        for i in tb_np:
            empty_list.append(Tensor(i))
        return tuple(empty_list)

    def construct(self, preds, labels):


        pred = self.get_pred(Tensor(preds), Tensor(self.mask))
        tem_label = self.get_pred(Tensor(labels), Tensor(self.mask))
        label = self.get_label(Tensor(tem_label))

        pred = np.array(pred)
        label = np.array(label)

        tem = np.stack(
            [np.equal(label,i) for i in range(self.num_classes)],1
        )
        Y = np.squeeze(tem).astype(float)
        N = Y.sum(0)

        loss = self.l2_loss(self.param)
        loss += self.sceloss(preds, labels)[0]
        mask = self.cast(self.mask, mstype.float32)
        mask_reduce = self.mean(mask)
        mask = mask / mask_reduce
        loss = loss * mask
        loss = self.mean(loss)

        for i in range(1):
            for j in range(1):
                if i!=j:
                    i_pred_pos = pred[ np.array( Tensor(Y[:,i], dtype=ms.int32) ),:][:,i]
                    i_pred_neg = pred[ np.array( Tensor(Y[:,j], dtype=ms.int32) ),:][:,i]

                    broadcast_to_0 = ops.BroadcastTo((i_pred_pos.shape[0],i_pred_neg.shape[0]))
                    i_pred_pos_expand = broadcast_to_0(self.expand_dims(i_pred_pos,1))
                    i_pred_pos_sub_neg = i_pred_pos_expand-i_pred_neg
                    ij_loss = self.exp(-self.gamma * i_pred_pos_sub_neg)
                    
                    i_pred_pos_index = ops.nonzero(Tensor(Y[:,i])).view(-1)
                    i_pred_neg_index = ops.nonzero(Tensor(Y[:,j])).view(-1)

                    i_pred_pos_adj = self.adj_maxtrix[np.array(Tensor(self.mask,dtype=ms.int32))][i_pred_pos_index]
                    i_pred_neg_adj = self.adj_maxtrix[np.array(Tensor(self.mask,dtype=ms.int32))][i_pred_neg_index]
                    i_pred_neg_self_adj =self.adj_self_matrix[np.array(Tensor(self.mask,dtype=ms.int32))][i_pred_neg_index]
                    broadcast_to_1 = ops.BroadcastTo((i_pred_pos_adj.shape[0],i_pred_neg_adj.shape[0],i_pred_pos_adj.shape[1]))
                    i_pred_pos_adj_expand = broadcast_to_1(self.expand_dims(i_pred_pos_adj,1))
                    
                    sub_ner = (np.logical_xor(i_pred_pos_adj_expand, np.logical_and(i_pred_pos_adj_expand,i_pred_neg_self_adj) ))
                    inter_ner = np.logical_and(i_pred_pos_adj_expand,i_pred_neg_adj)

                    num_inter_nonzero = np.count_nonzero(Tensor(inter_ner,dtype=ms.int32))
                    num_sub_nonzero = np.count_nonzero(Tensor(sub_ner,dtype=ms.int32))

                    if num_inter_nonzero>0 and num_sub_nonzero>0:
                        # sub
                        nonzero = ops.NonZero()
                        sub_ner_nonzero = self.nonzero_tuple(sub_ner)

                        a = self.cast_op(sub_ner_nonzero[0],ms.int32)
                        b = self.cast_op(sub_ner_nonzero[1],ms.int32)
                        I_sub_tem = self.concat_op((a,b))
                        I_sub = self.reshape(I_sub_tem, (2,-1))
                        I_sub = I_sub.T
                        V_sub = self.sigmoid( self.cast_op(sub_ner_nonzero[2], ms.float32))

                        S_sub = ms.COOTensor(Tensor(I_sub,dtype=ms.int64), V_sub,i_pred_pos_sub_neg.shape)
                        vi_sub = S_sub.to_dense()

                        # inter
                        inter_ner_nonzero = self.nonzero_tuple(inter_ner)
                        a = self.cast_op(inter_ner_nonzero[0],ms.int32) 
                        b = self.cast_op(inter_ner_nonzero[1],ms.int32)
                        I_inter_tem = self.concat_op((a,b))
                        I_inter = self.reshape(I_inter_tem,(2,-1))
                        I_inter = I_inter.T
                        V_inter = self.sigmoid( self.cast_op(inter_ner_nonzero[2], ms.float32))

                        S_inter = ms.COOTensor(Tensor(I_inter,dtype=ms.int64), V_inter,i_pred_pos_sub_neg.shape)
                        vi_inter = S_inter.to_dense()


                        vl_i = self.sigmoid((1+vi_sub)/(1+vi_inter))
                        v_i = 1-vl_i
                        ij_loss = self.reduce_sum( (1/(N[i]*N[j])*v_i*ij_loss) )*self.per
                        loss+=Tensor(ij_loss)
        return loss