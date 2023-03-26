"""
network config setting, will be used in train.py
"""

class ConfigGCN():
    """Configuration for GCN"""
    learning_rate = 0.001
    epochs = 50
    hidden1 = 64
    hidden2 = 64
    layer_num = 3
    dropout = 0.2
    weight_decay = 5e-4
    early_stopping = 50
    save_ckpt_steps = 50
    keep_ckpt_max = 10
    ckpt_dir = './ckpt'
    best_ckpt_dir = './best_ckpt'
    best_ckpt_name = 'best.ckpt'
    eval_start_epoch = 30
    save_best_ckpt = True
    eval_interval = 1
    dataset_name = 'cora'