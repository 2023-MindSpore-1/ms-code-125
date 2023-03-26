import os
import numpy as np
from mindspore import Tensor
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
from mindspore import Model, context

from src.gcn import StandGCN3, StandGCN2, StandGCN1
from src.config import ConfigGCN
from src.e_loss_fn import e_loss_FN
from src.dataset import get_adj_features_labels, load_processed_data_info
from model_utils.config import config as default_args

def get_gcn_net(config, input_dim, class_num, adj):
    if config.layer_num==1:
        gcn_net = StandGCN1(config, input_dim, class_num, adj)
    if config.layer_num==2:
        gcn_net = StandGCN2(config, input_dim, class_num, adj)
    if config.layer_num==3:
        gcn_net = StandGCN3(config, input_dim, class_num, adj)
    return gcn_net

def run_gpu_train():

    config = ConfigGCN()

    # get_dataset
    adj, feature, label_onehot, _ = get_adj_features_labels(default_args.data_dir)
    feature_d = np.expand_dims(feature, axis=0)
    label_onehot_d = np.expand_dims(label_onehot, axis=0)
    data = {"feature": feature_d, "label": label_onehot_d}
    dataset = ds.NumpySlicesDataset(data=data)

    # get_info
    num_nodes, num_classes, adj_bool, gem, gpr, train_mask, eval_mask, test_mask = load_processed_data_info(config.dataset_name)
    
    # get_model
    input_dim = feature.shape[1]
    class_num = label_onehot.shape[1]
    adj = Tensor(adj, dtype=mstype.float32)
    gcn_net = get_gcn_net(config, input_dim, class_num, adj)

    # get_ckpoint_cb
    ckpt_config = CheckpointConfig(save_checkpoint_steps=config.save_ckpt_steps,keep_checkpoint_max=config.keep_ckpt_max)
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_gcn',directory=config.ckpt_dir,config=ckpt_config)
    
    # get_optimizer
    optm = nn.Adam(gcn_net.trainable_params(), learning_rate = config.learning_rate)

    criterion = e_loss_FN(num_classes=num_classes, num_nodes=num_nodes, adj_maxtrix=adj_bool, global_effect_matrix=gem, global_perclass_mean_effect_matrix=gpr, mask=eval_mask, param=gcn_net.trainable_params()[0])
    model = Model(gcn_net, loss_fn = criterion, optimizer=optm, amp_level="O3")
    cb = [TimeMonitor(), LossMonitor(), ckpoint_cb]

    if default_args.train_with_eval:
        GCN_metric = GCNAccuracy(eval_mask)
        eval_model = Model(gcn_net, loss_fn = criterion, metrics={"GCNAccuracy": GCN_metric})
        eval_param_dict = {"model": eval_model, "dataset": dataset, "metrics_name": "GCNAccuracy"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval, eval_start_epoch=default_args.eval_start_epoch, save_best_ckpt = config.save_best_ckpt, ckpt_directory=config.best_ckpt_dir, best_ckpt_name = config.best_ckpt_name, metrics_name="GCNAccuracy")
        cb.append(eval_cb)

    model.train(config.epochs, dataset, callbacks=cb, dataset_sink_mode=True)

if __name__ == "__main__":
    print("==============================")
    if default_args.device_target != "GPU":
        print("Due to limitations of OPERATOR, please use GPU devices to run. If you own a GPU device, specify it in config.")
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        run_gpu_train()
    print("==============================")