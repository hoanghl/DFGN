# from multiprocessing import Pool
import multiprocessing
import threading
import json
import sys
import re
import os

from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer
from tqdm import tqdm
import spacy
import torch

from modules.utils import save_object, check_file_existence, ParallelHelper
from configs import args, logging, configs

if args.working_place != "local":
    torch.multiprocessing.set_sharing_strategy('file_system')

BERT_PATH       = f"{args.init_path}/_pretrained/BERT/{args.bert_model}/"
BERT_TOKENIZER  = f"{args.init_path}/_pretrained/BERT/{args.bert_model}-vocab.txt"

try:
    BERT_tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=False)
except OSError:
    BERT_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
nlp             = spacy.load("en_core_web_sm")

TOKEN_INIT      = BERT_tokenizer.cls_token_id
TOKEN_SEP       = BERT_tokenizer.sep_token_id
TOKEN_PAD       = BERT_tokenizer.pad_token_id

RATIO_TRAIN_TEST = 0.8


class PreprocessingHelper:
    def __init__(self, data_point):
        ##############################
        ### Tokenization
        ##############################
        tokens = self.tokenizer(data_point['sentence'])


        ##############################
        ### Convert token to id
        ##############################
        tokens = BERT_tokenizer.convert_tokens_to_ids(tokens)

        ##############################
        ### Truncating and Padding
        ##############################
        tokens = [TOKEN_INIT] + tokens[:configs['MAX_SEQ_LEN_W'] - 2] + [TOKEN_SEP]
        tokens.extend([BERT_tokenizer.pad_token_id for _ in range(configs['MAX_SEQ_LEN_W'] - len(tokens))])

        ##############################
        ### Create attention mask
        ##############################
        attn_mask = [int(tok > 0) for tok in tokens]

        self.preprocessed_datapoint = {
            '_id'           : data_point['_id'],
            '_id_context'   : data_point['_id_context'],
            'sentence'      : tokens,
            'attn_mask'     : attn_mask,
            'score'         : data_point['score']
        }


    def tokenizer(self, sentence):
        doc = nlp(sentence)
        # for token in doc.sentences[0].tokens:
        #     print(token.text)

        # tokens = BERT_tokenizer.tokenize(sentence)
        tokens = [token.text for token in doc]

        return tokens


def f_kernel(data: list, queue):
    for data_point in data:
        queue.put(PreprocessingHelper(data_point).preprocessed_datapoint)


class CustomizedDataset(Dataset):
    def __init__(self, dataset_raw):
        self.dataset = ParallelHelper(f_kernel, dataset_raw, args.n_cpus).launch()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        return self.dataset[item]


#############################################################################################
# Function to read data from original JSON files, preprocess, create Dataset
# amd store them into 'pickle' file
#############################################################################################
def generate_train_datasets():
    """
    Create Dataset objects (train, dev, test) from original
    JSON dataset files
    """

    path_data_train = f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/data_train.json"
    path_data_dev   = f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/data_dev.json"

    ###################
    ## Check whether files "data_dev.json" and "data_train.json"
    ## have been created
    ###################
    if not check_file_existence(path_data_train) or \
            not check_file_existence(path_data_dev):
        ## If those files have not been created, create them first

        PATH = f"{args.init_path}/_data/QA/HotpotQA/"
        FILES = [
            ("hotpot_train_v1.1.json",      "data_train.json"),
            ("hotpot_dev_fullwiki_v1.json", "data_dev.json")
        ]

        for file, store_file in FILES:
            path_data = f"{PATH}{file}"

            ############################
            ## Đọc data, tạo thành các sample
            ############################
            logging.info(f"{file} - Read file and create samples")

            if not check_file_existence(path_data):
                raise ValueError(f"Path {path_data} not existed.")

            samples = []
            with open(path_data, 'r') as dat_file:
                for datapoint in json.load(dat_file):
                    contexts = []
                    for nth_context, context in enumerate(datapoint['context']):
                        samples.append({
                            '_id'           : datapoint['_id'],
                            '_id_context'   : nth_context,
                            'question'      : datapoint['question'],
                            'support_fact'  : [fact[0] for fact in datapoint['supporting_facts']],
                            'context'       : ''.join(context[1]),
                            'score'         : 0
                        })

            ############################
            ## Sử dụng IR để phân loại
            ############################
            logging.info(f"{file} - Use naive IR to score samples")

            def f_thread(samples, up_bound, low_bound):
                # print(f"{up_bound} - {low_bound}")
                for sample in tqdm(samples[up_bound:low_bound]):
                    for fact in sample['support_fact']:
                        try:
                            if len(re.findall(f"{re.escape(fact)}", sample['context'])) > 0:
                                sample['score'] = 1
                                break
                        except Exception:
                            print(fact)
                            sys.exit()

            n_samples = len(samples)
            threads = []
            for i in range(4):
                threads.append(threading.Thread(target=f_thread, args=(samples,
                                                                       n_samples * i // 4,
                                                                       n_samples * (i + 1) // 4)))
            for thread_ in threads:
                thread_.start()

            for thread_ in threads:
                thread_.join()

            ### Save what has been processed into JSON file
            path = f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/{store_file}"
            try:
                os.makedirs(os.path.dirname(path))
            except FileExistsError:
                logging.warning(f"=> Folder {os.path.dirname(path)} exists.")
            with open(path, "w+") as res_file:
                json.dump(samples, res_file, ensure_ascii=False, indent=2)

    else:
        ##############################
        ### Read data and do preprocessing
        ##############################
        logging.info("1. Read data file")
        with open(path_data_train, 'r') as dat_file:
            data_raw_train = [
                {
                    '_id'           : x['_id'],
                    '_id_context'   : x['_id_context'],
                    'sentence'      : x['question'] + " " + x['context'],
                    'score'         : x['score']
                } for x in json.load(dat_file)
            ]
        with open(path_data_dev, 'r') as dat_file:
            data_raw_dev = [
                {
                    '_id'           : x['_id'],
                    '_id_context'   : x['_id_context'],
                    'sentence'      : x['question'] + " " + x['context'],
                    'score'         : x['score']
                } for x in json.load(dat_file)
            ]


        ##############################
        ### Create dataset
        ##############################
        logging.info("2. Create dataset")

        ### Create Dataset object
        dataset_dev     = CustomizedDataset(data_raw_dev)
        save_object("./backup_files/select_paras/dataset_dev.pkl.gz", dataset_dev)
        logging.info("Successfully created dataset dev.")

        dataset_train   = CustomizedDataset(data_raw_train)

        ## Split training set into training and testing
        length_train    = int(RATIO_TRAIN_TEST * len(data_raw_train))
        length_test     = len(data_raw_train) - length_train
        dataset_train, dataset_test = random_split(dataset_train, [length_train, length_test])

        ### Back up pickle file
        save_object(f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/dataset_train.pkl.gz", dataset_train)
        save_object(f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/dataset_test.pkl.gz", dataset_test)
        logging.info("Successfully store into pickle files.")


if __name__ == "__main__":
    generate_train_datasets()
