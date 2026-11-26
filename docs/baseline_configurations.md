# Experiment configuration and baseline reference index

This document distinguishes the manuscript's described evaluation protocol, the configuration files currently distributed in this repository, and external baseline reference implementations. Upstream defaults are not a record of the exact settings used to generate the reported comparison values. For reproducibility of a specific reproduced result, the effective run configuration, code revision and any overrides are required.

## 1. Conventional benchmark protocol described in the manuscript

| Setting | Description |
| --- | --- |
| Datasets | PEMS03, PEMS04, PEMS07, PEMS08, METR-LA, PEMS-BAY |
| Sampling interval | 5 minutes |
| Input / forecast length | 12 / 12 steps |
| Chronological split (train / validation / test) | 60% / 20% / 20% |
| Normalization | Z-score; training-set statistics only |
| Metrics | MAE, RMSE, MAPE |
| MVST-DSSL random seeds | 1, 2, 3; mean across independent runs |
| MVST-DSSL checkpoint selection | Lowest validation MAE |
| MVST-DSSL maximum epochs / patience / min improvement | 80 / 15 / 0.0001 |
| MVST-DSSL optimizer / initial LR / weight decay | Adam / 0.001 / 0.0015 |
| MVST-DSSL LR schedule | MultiStepLR, milestones 25, 45, 65 |
| MVST-DSSL batch size | 8 for PEMS07; 16 otherwise |
| MVST-DSSL embeddings (input, time-of-day, day-of-week, adaptive) | 24, 24, 24, 100 |
| MVST-DSSL heads / Transformer layers / FFN width / dropout | 4 / 2 / 64 / 0.1 |
| MVST-DSSL mask ratio / contrastive temperature | 0.1 / 0.1 |
| MVST-DSSL graph perturbation candidates | 0.05, 0.10, 0.15, 0.20, 0.25 (selection by validation) |
| MVST-DSSL reconstruction / contrastive loss weights | 0.05 / 0.1 |

The MVST-DSSL-specific optimizer and architecture settings above are not asserted to be shared by all baselines.

## 2. Configuration files in this repository

| Dataset | Model configuration | Data preparation |
| --- | --- | --- |
| PEMS03 | [PEMS03.py](../baselines/MVST-DSSL/PEMS03.py) | [script](../scripts/data_preparation/PEMS03/generate_training_data.py) |
| PEMS04 | [PEMS04.py](../baselines/MVST-DSSL/PEMS04.py) | [script](../scripts/data_preparation/PEMS04/generate_training_data.py) |
| PEMS07 | [PEMS07.py](../baselines/MVST-DSSL/PEMS07.py) | [script](../scripts/data_preparation/PEMS07/generate_training_data.py) |
| PEMS08 | [PEMS08.py](../baselines/MVST-DSSL/PEMS08.py) | [script](../scripts/data_preparation/PEMS08/generate_training_data.py) |
| METR-LA | No dataset-specific model configuration in this repository | [script](../scripts/data_preparation/METR-LA/generate_training_data.py) |
| PEMS-BAY | No dataset-specific model configuration in this repository | [script](../scripts/data_preparation/PEMS-BAY/generate_training_data.py) |

Model source: [architecture](../baselines/MVST-DSSL/arch/MVST-DSSL.py). Training entry point: [experiments/train.py](../experiments/train.py); its explicit configuration argument is `--cfg` (or `-c`), and GPU argument is `--gpus`.

The distributed Python files are reference configurations, not independently validated archived run manifests. In particular, the four PeMS files set FFN width to 256; PEMS07 sets three layers; PEMS08 sets training batch size to 8 and mask ratio to 0.8. These values differ from the manuscript-described protocol in Section 1. The PEMS08 file also contains machine-specific data paths. The exact effective settings of published runs should be established from their original records before reproducing or attributing a numerical result. Configuration values have not been overwritten to manufacture agreement with the manuscript.

## 3. External baseline configuration references

The links below point to upstream implementations, example configuration files or paper descriptions for the conventional comparison set. They are reference resources, not claims that every model was rerun using the linked defaults. A paper-reported result and a locally reproduced result must be distinguished at the dataset/result level. A dash indicates that an exact configuration reference has not been established here.

| Model | Upstream reference / configuration entry |
| --- | --- |
| HA | [DCRNN paper, Appendix E](https://arxiv.org/abs/1707.01926): seasonal historical average; exact seasonal window of the local comparison not established |
| ARIMA | [DCRNN paper, Appendix E](https://arxiv.org/abs/1707.01926): ARIMA reference variant; actual local order not established |
| VAR | [DCRNN paper, Appendix E](https://arxiv.org/abs/1707.01926): VAR reference variant; actual local lag order not established |
| DCRNN | [METR-LA YAML](https://github.com/liyaguang/DCRNN/blob/master/data/model/dcrnn_la.yaml), [PEMS-BAY YAML](https://github.com/liyaguang/DCRNN/blob/master/data/model/dcrnn_bay.yaml) |
| STGCN | [upstream main.py](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/main.py) |
| ASTGCN | [PEMS04/08 configurations](https://github.com/wanhuaiyu/ASTGCN/tree/master/configurations) |
| GWNet | [upstream training arguments](https://github.com/nnzhan/Graph-WaveNet/blob/master/train.py) |
| STSGCN | [archived dataset configuration directories](https://github.com/Davidham3/STSGCN/tree/master/archive/mxnet/config) |
| STFGNN | [dataset configuration directories](https://github.com/MengzhangLI/STFGNN/tree/master/config) |
| STGODE | [upstream implementation](https://github.com/square-coder/STGODE) |
| STG-NCDE | [PEMSD4 configuration](https://github.com/jeongwhanchoi/STG-NCDE/blob/main/model/PEMSD4_GCDE.conf) |
| DSTAGNN | [PEMS03/04/07/08 configuration files](https://github.com/SYLan2019/DSTAGNN/tree/main/configurations) |
| STAEformer | [dataset YAML](https://github.com/XDZhelheim/STAEformer/blob/main/model/STAEformer.yaml) |
| GRAM-ODE | [upstream implementation](https://github.com/zbliu98/GRAM-ODE) |
| PDFormer | [PEMS08 dataset settings](https://github.com/BUAABIGSCity/PDFormer/blob/master/PeMS08.json), [model defaults](https://github.com/BUAABIGSCity/PDFormer/blob/master/libcity/config/model/traffic_state_pred/PDFormer.json) |
| Auto-DSTSGN | [architecture JSON](https://github.com/jinguangyin/Auto-DSTSGN/blob/main/config/individual_3layer_12T.json), [run examples](https://github.com/jinguangyin/Auto-DSTSGN) |
| STPGNN | [configuration](https://github.com/Kongwy5689/STPGNN/blob/main/data/config.json), [training entry](https://github.com/Kongwy5689/STPGNN/blob/main/train.py) |
| ADM-STNODE | Original paper cited in the manuscript; exact configuration not established |
| MVSTL | Original paper cited in the manuscript; exact configuration not established |
| AutoSTF | [six dataset YAML files](https://github.com/tfeilyu/AutoSTF/tree/main/model_settings) |
| MAGCRN | [dataset-specific MAGCRN configuration files](https://github.com/Aazeb/MAGCRN/tree/main/model) |
| LightST | [upstream implementation and dataset configuration directory](https://github.com/lizzyhku/TP/tree/main/model) |
| HyperD | [training entry](https://github.com/ll121202/HyperD/blob/master/train.py), [upstream repository](https://github.com/ll121202/HyperD) |

Additional benchmark reference: [BasicTS](https://github.com/GestaltCogTeam/BasicTS). Its current version should not be assumed identical to the version used by a historical run.

## 4. Reproduction record schema

For each locally reproduced model-dataset combination, the run manifest should identify:

- model, dataset, result origin (local reproduction or published result), implementation URL and immutable revision;
- effective configuration file and execution command, including command-line overrides;
- chronological split, scaler fitting scope, input and output lengths, metric aggregation and null-value handling;
- architecture-specific settings (e.g., hidden channels, layer count, graph order, attention heads);
- optimizer, initial learning rate, learning-rate schedule, weight decay, batch size, epoch budget, early stopping and checkpoint criterion;
- random seeds, aggregation across runs and checkpoint/test record identifiers.

For a published result, record the paper, table and dataset-specific protocol instead of inventing a local training configuration. Upstream reference defaults must be distinguished from effective run parameters.

## 5. LargeST reference protocol

The manuscript describes SD (716 nodes) and GBA (2,352 nodes) using 2019 data at 15-minute resolution, chronological 6:2:2 splits, 12 input and 12 prediction steps, and MAE/RMSE/MAPE averaged over the forecast horizons. Data preparation entry points: [SD](../scripts/data_preparation/SD/generate_training_data.py), [GBA](../scripts/data_preparation/GBA/generate_training_data.py).

Reference results are attributed to the [LargeST benchmark](https://github.com/liuxu77/LargeST) for DCRNN, STGCN, ASTGCN, GWNet, STGODE and DSTAGNN; the MAGE paper for STAEformer; and the respective papers for PatchSTG, MAGE and RAST. See the manuscript appendix for the source attribution and the qualification concerning RAST's split descriptions. These reference results are not represented as local baseline reruns.

- [LargeST paper](https://arxiv.org/abs/2306.08259)
- [PatchSTG paper](https://arxiv.org/abs/2412.09972)
- [MAGE paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/54c9bfb0885ae07f23607f617ab64c2b-Abstract-Conference.html)
- [RAST paper](https://arxiv.org/abs/2508.16623)

This index supports locating reference settings; it does not substitute for the effective experiment manifests or independently validate the reported results.
