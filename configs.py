'''
This file processes arguments and configs
'''
import argparse
import logging

import torch


###############################
# Đọc các argument từ CLI
###############################
parser = argparse.ArgumentParser()

parser.add_argument("--working_place", choices=['local', 'rtx', 'dgx'],
                    help="working environment", default='local')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_cpus", type=int, default=4)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                    help="default pretrain BERT model")
parser.add_argument("--max_seq_length", type=int, default=512,
                    help="Max length for each sequence, max length equal max length of BERT, more than this number"
                         "will be truncated, less will be padded.")
parser.add_argument("--task", type=str, help="Select task to do")
parser.add_argument("--device", type=str, default="default", choices=["default", "cpu", "cuda"],
                    help="Select device to run")

# args = parser.parse_args()
args, _ = parser.parse_known_args()


###############################
# Thêm các argument
###############################
## args = working_place
if args.working_place == "local":
    args.init_path  = "/Users/hoangle/Projects/VinAI"
elif args.working_place == "rtx":
    args.init_path  = "/home/ubuntu/hoanglh88/works"
elif args.working_place == "dgx":
    args.init_path  = "/home/ubuntu"

## args = bert_model
if args.bert_model == "":
    args.bert_model = f"{args.init_path}/_pretrained/BERT/bert-base-uncased/"

## args = device
if args.device == "default":
    args.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    args.device = torch.device(args.device)


###############################
# Cấu hình logging
###############################
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().setLevel(logging.INFO)


###############################
# Cấu hình các biến config
###############################
configs = {
    ## threshold used to select golden passages
    'DELTA' : 0.1,
}