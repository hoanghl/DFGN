import json

from modules.utils import save_object, check_file_existence
from configs import args, logging, configs


def encode_query_context():
    """
    Read file `selected_query_context.json`, do encoding
    and save result into `encoded_features.pkl.gz`
    """

    ###################################
    ## Read file `selected_query_context.json`
    ###################################
    ## Check file existence first, then read
    if not check_file_existence(args.path_selectedQueryContext):
        logging.error(f"Err: File {args.path_selectedQueryContext} not found.")
        raise FileNotFoundError

    with open(args.path_selectedQueryContext, 'r') as dat_file:
        data = json.load(dat_file)

    ## Preprocess and create tensors
    # {
    #     'query' : "Why ?",
    #     'context': [
    #         "The nasm",
    #         "fds"
    #     ]
    # }

    ###################################
    ## Pass through models
    ###################################
    ## There are 2 models to be passed:
    ## pretrained BERT and BiDAF

    ###################################
    ## Save into `encoded_features.pkl.gz`
    ###################################
