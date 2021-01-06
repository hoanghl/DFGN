from operator import itemgetter
import json

from torch.utils.data import Dataset, DataLoader
import pandas as pd

from modules.utils import save_object, load_object, check_file_existence
from modules.encoding.utils import sentence2tokens
from configs import args, logging, configs


class CustomDataset(Dataset):
    def __init__(self, dataset):
        super(CustomDataset, self).__init__()

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


def get_items_by_indx(list_items: list, indx: list) -> list:
    """
    Get items in `list_items` by list of `indx`
    Args:
        list_items (list): list of items
        indx (list): list of indices

    Returns:
        list of items
    """
    return itemgetter(*indx)(list_items)


def encode_query_context(threshold: float):
    """
    Read file `selected_query_context.json`, do encoding
    and save result into `encoded_features.pkl.gz`
    Args:
        threshold ():

    Returns:

    """

    ###################################
    ## Read original JSON files and saved pkl.gz file
    ## to form data points
    ###################################
    path_original_train = ""
    path_original_dev   = ""
    path_saved_train    = ""
    path_saved_dev      = ""

    ## Check file existence first
    for path_ in [path_original_train, path_original_dev,
                  path_saved_train, path_saved_dev]:
        if not check_file_existence(path_):
            logging.error(f"Err: File {path_} not found.")
            raise FileNotFoundError

    ## Read files
    with open(path_original_train, 'r') as dat1_file, open(path_original_dev, 'r') as dat2_file:
        original_train  = json.load(dat1_file)
        original_dev    = json.load(dat2_file)

    saved_train     = pd.DataFrame(load_object(path_saved_train))
    saved_dev       = pd.DataFrame(load_object(path_saved_dev))

    ## Loop entire original datasets (original_train, original_dev)
    ## filter

    ## Hai biến sau đây được dùng để chứa data cho training và testing của DFGN
    ## nhưng trước đó chúng sẽ được đưa qua BERT và BiDAF

    def get_data_for_encoding(data, saved_data):
        results = list()

        for datapoint in data:
            ### Lọc ra các golden passage
            filtered        = saved_data[(saved_data['id'] == datapoint['_id']) &
                                         (saved_data['predicted_score'] >= threshold)]['_id_context']

            selected_paras  = get_items_by_indx(datapoint['context'], filtered)

            ### Ghép các golden passage lại thành một đoạn context duy nhất
            context = ""
            for para in selected_paras:
                context = context + ''.join(para[1]) + ' '


            ### Append vào list 'results'
            results.append({
                '_id'       : datapoint['_id'],
                'c_plain'   : context,
                'c_word'    : sentence2tokens(context),
                'c_char'    : None,
                'q_plain'   : datapoint['question'],
                'q_word'    : sentence2tokens(datapoint['question']),
                'q_char'    : None,
                'answer'    : datapoint['answer']
            })

        return results

    dataset_train   = get_data_for_encoding(original_train, saved_train)
    dataset_dev     = get_data_for_encoding(original_dev, saved_dev)

    ## Back up 2 files 'dataset_train' and 'dataset_dev'
    save_object("./backup_files/encoding/dataset_train.pkl.gz", dataset_train)
    save_object("./backup_files/encoding/dataset_dev.pkl.gz", dataset_dev)


    ###################################
    ## Pass through models
    ###################################
    ## Create dataset
    iterator_train   = DataLoader(CustomDataset(dataset_train),
                                  batch_size=args.batch_size)
    iterator_dev     = DataLoader(CustomDataset(dataset_dev),
                                  batch_size=args.batch_size)

    ## Let data pass through BERT


    ## Let data pass through BiDAF

    ###################################
    ## Save into `encoded_features.pkl.gz`
    ###################################


if __name__ == '__main__':
    pass