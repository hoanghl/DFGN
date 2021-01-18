#!/bin/bash
## Template created by Hoang Le
## Dec 2, 2020


WORKING_PLACE="dgx"
CONDA_ENV_NAME="pytorch"
PROJECT_NAME="DFGN"
DATA_FILE="data/QA/HotpotQA.tar.gz"
PRETRAINED_FILE="pretrained/BERT.tar.gz"

if [ $WORKING_PLACE = "dgx" ]
then
    INIT_PATH_ISILON="/mnt/vinai"
elif [ $WORKING_PLACE = "rtx" ]; then
    INIT_PATH_ISILON="/vinai/hoanglh88"
else
    echo "Variable WORKING_PLACE invalid."
    return 1
fi


echo "=============================="
echo "===== SCRIPT for Tessera ====="
echo "=============================="
echo "Working place:" $WORKING_PLACE
NOW=$(date +"%b-%d-%Y %r")
echo "Start time   :" $NOW

## By default, pwd is "/root"
echo "0. Do some miscellaneous things"
echo "0.1. Create working directory"
mkdir -p _data
mkdir -p _pretrained

echo "0.2. Activate conda and environment"
export PATH="/opt/miniconda3/bin:$PATH"
source activate $INIT_PATH_ISILON/conda_env/$CONDA_ENV_NAME


echo "1. Extract stuffs from Isilon"
## By default, data in Isilon is mounted at "$INIT_PATH_ISILON/"

echo "1.1. Extract data"
tar -xzvf $INIT_PATH_ISILON/"$DATA_FILE" -C _data
echo "1.2. Extract pretrained models such as BERT, GloVe..."
tar -xzvf $INIT_PATH_ISILON/"$PRETRAINED_FILE" -C _pretrained
echo "1.3. Extract previous work(s)"
tar -xzvf $INIT_PATH_ISILON/$PROJECT_NAME.tar.gz -C .


echo "2. Do main work"
cd $PROJECT_NAME || return

## Enter the code you want to run here
CUDA_VISIBLE_DEVICES=0,1 python -m modules.para_selection.para_selector --batch_size 16 --working_place $WORKING_PLACE --task selectparas_train --n_cpus 4

echo "3. Wrap up everything"
tar -czvf $PROJECT_NAME.tar.gz $PROJECT_NAME
mv $PROJECT_NAME.tar.gz $INIT_PATH_ISILON/$PROJECT_NAME.tar.gz