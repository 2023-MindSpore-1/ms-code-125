import argparse
from distutils.command.config import config
from logging import critical
import numpy as np
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint
from mindspore import Tensor
from mindspore import Model, context

from src.config import ConfigGCN
from src.dataset import get_adj_features_labels
from src.e_loss_fn import e_loss_FN
from src.gcn import StandGCN1, StandGCN2, StandGCN3

from src.dataset import get_adj_features_labels, load_processed_data_info

def get_gcn_net(config, input_dim, class_num, adj):
    if config.layer_num==1:
        gcn_net_test = StandGCN1(config, input_dim, class_num, adj)
    if config.layer_num==2:
        gcn_net_test = StandGCN2(config, input_dim, class_num, adj)
    if config.layer_num==3:
        gcn_net_test = StandGCN3(config, input_dim, class_num, adj)
    return gcn_net_test

def run_gcn_infer():

    config = ConfigGCN()
    data_name = config.dataset_name
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--data_dir", type=str, default="./data/"+str(data_name)+"/"+str(data_name)+"_mr")
    parser.add_argument("--model_ckpt", type=str, required=True,
                        help="existed checkpoint address.")
    args_opt = parser.parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", save_graphs=False)
    adj, feature, label_onehot, _ = get_adj_features_labels(args_opt.data_dir)
    feature_d = np.expand_dims(feature, axis=0)
    label_onehot_d = np.expand_dims(label_onehot, axis=0)
    data = {"feature": feature_d, "label": label_onehot_d}
    dataset = ds.NumpySlicesDataset(data=data)
    input_dim = feature.shape[1]
    class_num = label_onehot.shape[1]
    adj = Tensor(adj, dtype=mstype.float32)
    gcn_net_test = get_gcn_net(config, input_dim, class_num, adj)
    gcn_net_test.set_train(False)
    load_checkpoint(args_opt.model_ckpt, net=gcn_net_test)
    num_nodes, num_classes, adj_bool, gem, gpr, train_mask, eval_mask, test_mask = load_processed_data_info(config.dataset_name)

    criterion = e_loss_FN(num_classes=num_classes, num_nodes=num_nodes, adj_maxtrix=adj_bool, global_effect_matrix=gem, global_perclass_mean_effect_matrix=gpr, mask=eval_mask, param=gcn_net_test.trainable_params()[0])
    eval_metrics = {'Acc': nn.Accuracy()}
    model = Model(gcn_net_test, loss_fn = criterion, metrics=eval_metrics)
    res = model.eval(dataset, dataset_sink_mode=True)
    print(res)

if __name__ == "__main__":
    print("++++++++++++++++++++++++++++++")
    run_gcn_infer()
    print("++++++++++++++++++++++++++++++")