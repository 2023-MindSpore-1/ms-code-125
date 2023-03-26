import numpy as np
import scipy.sparse as sp
import mindspore.dataset as ds
import mindspore as ms
from mindspore import Tensor, ops
from mindspore.dataset.engine import OutputFormat

def index2adj_bool(edge_index, nnode=2708):
    indx = edge_index
    adj = np.zeros((nnode,nnode),dtype = 'bool')
    adj[(indx[0],indx[1])]=1
    new_adj = Tensor(adj)
    return new_adj

def index2dense(edge_index, nnode=2708):
    indx = edge_index#.asnumpy()
    adj = np.zeros((nnode,nnode),dtype = "int8")
    adj[(indx[0],indx[1])]=1
    new_adj = Tensor.from_numpy(adj)#.float()
    return new_adj

def get_step_split(all_nodes_idx,all_label,nclass=7):
    base_valid_each = 30
    imb_ratio = 5.0
    head_list = [i for i in range(nclass//2)]

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    base_train_each = int( len(all_nodes_idx) * 0.05 / (t_num + h_num * imb_ratio) )

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list: 
        idx2train[i_h] = int(base_train_each * imb_ratio)
        idx2valid[i_h] = int(base_valid_each * 1) 

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list: 
        idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []

    for iter1 in all_nodes_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:break

    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_nodes_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==total_valid_size:break

    assert sum(valid_list)==total_valid_size
    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx,valid_idx,test_idx,train_node

def get_split(all_nodes_idx,all_label,nclass = 10):
    train_each = 20
    valid_each = 30
    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []
    for iter1 in all_nodes_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < train_each:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)
        if sum(train_list)==train_each*nclass:break
    assert sum(train_list)==train_each*nclass
    after_train_idx = list(set(all_nodes_idx)-set(train_idx))
    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < valid_each:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==valid_each*nclass:break
    assert sum(valid_list)==valid_each*nclass
    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx,valid_idx,test_idx,train_node

def load_processed_data_info(data_name):
    data_file = "data/"+str(data_name)+"/"+str(data_name)+"_mr"
    target_data = ds.GraphData(data_file)

    # get_num_nodes
    nodes = target_data.get_all_nodes(0)
    node_idx_list = nodes.tolist()
    target_data.num_nodes = len(node_idx_list)

    # get_num_classes    
    raw_tensor = target_data.get_node_feature(node_idx_list, [1,2])
    #target_data.num_features = raw_tensor[0].shape[1]
    target_data_y = Tensor(raw_tensor[1])
    target_data.y = target_data_y
    target_data.x = raw_tensor[0]
    target_data.num_classes = np.max(target_data_y.asnumpy())+1

    # get_adj_bool
    data = ds.GraphData(data_file)
    edge_index = edge_index = data.get_all_neighbors(node_idx_list, 0, output_format=OutputFormat.COO).T
    target_data.adj_bool=index2adj_bool(edge_index,target_data.num_nodes)

    # get_global_effect_matrix
    A     = index2dense(edge_index, len(target_data.get_all_nodes(0).tolist()))
    eye = ops.Eye()
    A_hat = A+ eye(A.shape[0], A.shape[0], ms.float64)
    A_hat_np = A_hat.asnumpy()
    D = np.diag(np.sum(A_hat_np,1))
    D = np.sqrt(np.linalg.inv(D))
    D = Tensor(D)
    matmul = ops.MatMul()
    A_hat = matmul(matmul(D, Tensor(A_hat)),D)
    target_data.gem = 0.15 * ( Tensor(np.linalg.inv(((eye(A.shape[0], A.shape[0], ms.float64)) - (1 - 0.15) * A_hat).asnumpy())) )
    
    # get_train_mask
    train_mask_list, valid_mask_list, test_mask_list, target_data.train_node = get_step_split(node_idx_list, target_data_y.asnumpy(), nclass=target_data.num_classes)
    target_data.train_mask = Tensor(np.zeros(len(node_idx_list), bool))
    target_data.valid_mask = Tensor(np.zeros(len(node_idx_list), bool))
    target_data.test_mask  = Tensor(np.zeros(len(node_idx_list), bool))
    tem_train_mask = target_data.train_mask.asnumpy()
    tem_valid_mask = target_data.valid_mask.asnumpy()
    tem_test_mask = target_data.test_mask.asnumpy()
    for i in train_mask_list:
        tem_train_mask[i] = True
    for i in valid_mask_list:
        tem_valid_mask[i] = True
    for i in test_mask_list:
        tem_test_mask[i] = True
    target_data.train_mask = Tensor(tem_train_mask)
    target_data.eval_mask = Tensor(tem_valid_mask)
    target_data.test_mask = Tensor(tem_test_mask)

    # get_global_perclass_mean_effect_matrix
    eye = ops.Eye()
    matmul = ops.MatMul()
    squeeze = ops.Squeeze()
    ReduceMean = ops.ReduceMean(keep_dims=True)
    transpose = ops.Transpose()
    gpr_matrix = []
    for iter_c in range(target_data.num_classes):
        tem_Pi = target_data.gem.asnumpy()
        tem_iter_Pi = tem_Pi[target_data.train_node[iter_c]]
        iter_Pi = Tensor(tem_iter_Pi)
        iter_gpr = ReduceMean(iter_Pi, 0)
        iter_gpr = squeeze(iter_gpr)
        gpr_matrix.append(iter_gpr)
    gpr_matrix_np = []
    for index in range(len(gpr_matrix)):
        gpr_matrix_np.append(gpr_matrix[index].asnumpy().astype(np.float32))
    temp_gpr = np.stack(gpr_matrix_np)
    temp_gpr = Tensor(temp_gpr)
    temp_gpr = transpose(temp_gpr,(0,1)).T
    target_data.gpr = temp_gpr

    return target_data.num_nodes, target_data.num_classes, target_data.adj_bool, target_data.gem, target_data.gpr, target_data.train_mask ,target_data.eval_mask, target_data.test_mask

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def get_adj_features_labels(data_dir):
    g = ds.GraphData(data_dir)
    nodes = g.get_all_nodes(0)
    nodes_list = nodes.tolist()
    row_tensor = g.get_node_feature(nodes_list, [1, 2])
    features = row_tensor[0]
    labels = row_tensor[1]

    nodes_num = labels.shape[0]
    class_num = labels.max() + 1
    labels_onehot = np.eye(nodes_num, class_num)[labels].astype(np.float32)

    neighbor = g.get_all_neighbors(nodes_list, 0)
    node_map = {node_id: index for index, node_id in enumerate(nodes_list)}
    adj = np.zeros([nodes_num, nodes_num], dtype=np.float32)
    for index, value in np.ndenumerate(neighbor):
        if value >= 0 and index[1] > 0:
            adj[node_map[neighbor[index[0], 0]], node_map[value]] = 1
    adj = sp.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) + sp.eye(nodes_num)
    nor_adj = normalize_adj(adj)
    nor_adj = np.array(nor_adj.todense())
    return nor_adj, features, labels_onehot, labels

