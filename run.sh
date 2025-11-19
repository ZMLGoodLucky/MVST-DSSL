#!/bin/bash
python experiments/train.py -c baselines/MVST_DSSL/PEMS04.py --gpus '0'
python experiments/train.py -c baselines/MVST_DSSL/PEMS08.py --gpus '0'
python experiments/train.py -c baselines/MVST_DSSL/PEMS03.py --gpus '0'
python experiments/train.py -c baselines/MVST_DSSL/PEMS07.py --gpus '0'
