#!/usr/bin/env bash

## Sample command: zsh launcher.sh rtx

BATCH_SIZE=16

## 1. Select paras
## 1.1. Tạo data
### Phân lập thành các paragraphs từ file JSON của dataset
#python -m modules.para_selection.DataHelper --batch_size 16 --working_place $1 --n_cpus 4

### Preprocess với các paragraphs, tạo thành các dataset sẵn sàng cho việc train
python -m modules.para_selection.DataHelper --batch_size $BATCH_SIZE --working_place $1 --n_cpus 4

## 1.2. Train paras selector
#python -m modules.para_selection.para_selector --batch_size $BATCH_SIZE --working_place $1 --task selectparas_train

## 1.3. Inference

## 2. Encode dataset and query