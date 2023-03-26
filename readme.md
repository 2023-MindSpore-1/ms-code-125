# TOPOAUC

An implement of the ACM MM 22 paper: A Unified Framework against Topology and Class Imbalance.

## Environments

* **Python** 3.9.15
* **Mindspore** 2.0.0 nightly
* **CUDA** 11.6
* **Ubuntu** 18.04

## Data:

Before the whole process, the dataset cora/citeseer/pubmed would be downloaded to `data\dataset_name\`  and Convert their data types as mindrecord.

You would download the datast in: [here](https://github.com/kimiyoung/planetoid/)

Then, through the [scripts](https://gitee.com/mindspore/models.git) provided by mindspore, they can be converted to mindrecord format.

For example as Cora:

```bash
!git clone https://gitee.com/mindspore/models.git
SRC_PATH = "./cora"
MINDRECORD_PATH = "./cora_mindrecord"

!rm -rf $MINDRECORD_PATH
!mkdir $MINDRECORD_PATH

!python models/utils/graph_to_mindrecord/writer.py --mindrecord_script cora --mindrecord_file "$MINDRECORD_PATH/cora_mr" --mindrecord_partitions 1 --mindrecord_header_size_by_bit 18 --mindrecord_page_size_by_bit 20 --graph_api_args "$SRC_PATH"
```

in the end, The dataset should be organized into the following form：

```
ms_GraphAUC-main
├─data
│  ├─cora
│  │  ├─cora_mr
│  │  └─cora_mr.db
│  │ 
│  ├─citeseer
│  │  ├─citeseer_mr
│  │  └─citeseer_mr.db
│  │ 
│  └─pubmed
│     ├─pubmed_mr
│     └─pubmed_mr.db
...
```

## Train:

When runing `train.py` for the default config, you would change the hyper-parameter by change `config.py`

## Eval:

You can run `python eval.py` to test the final  performance of the selected CKPT file.

