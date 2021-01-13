""" This file contains common functions for modules in
encoding query and context """

from operator import itemgetter

from transformers import BertTokenizer
import torch
import spacy

from configs import args, configs


BERT_PATH       = f"{args.init_path}/_pretrained/BERT/{args.bert_model}/"
BERT_TOKENIZER  = f"{args.init_path}/_pretrained/BERT/{args.bert_model}-vocab.txt"

BERT_tokenizer  = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=False)
nlp             = spacy.load("en_core_web_sm")



def get_entities_from_context(context: str) -> list:
    """
    Extract entities from every single sentence in context
    :param context: context to be segmented into sentences and find entities
    :type context: str
    :return: list containing entities and sentence they belong
    :rtype: list
    """
    import re
    import stanza

    def clean_entity(entity:str) -> str:
        """
        Clean 'entity'
        :param entity: entity to be cleaned
        :type entity: str
        :return: cleaned entity
        :rtype: str
        """

        ## Remove Possessive case 's
        entity = re.sub(r"\'s", "", entity)

        ## Remove double quote ""
        entity = re.sub(r"(\\)?(\"|\'|\'\')", "", entity)

        return entity


    stanza_res_path = f"{args.init_path}/_pretrained/stanza_resources"
    nlp             = stanza.Pipeline(lang='en', processors='tokenize,ner',
                                      dir=stanza_res_path, verbose=False)

    ############
    ## Do sentence segmentation and NER
    ############
    doc = nlp(context)

    ## Prepare variable 'result' to store entities and
    ## their corresponding sentence
    results = list()

    for sentence in doc.sentences:
        for entity in nlp(sentence.text).ents:
            results.append({
                'entity'    : clean_entity(entity.text),
                'span'      : sentence.text
            })

    return results


def sentence2tokens(sentence: str) -> object:
    """
    Tokenize 'sentence', do truncating/padding and
    Args:
        sentence (str): Sentence to be tokenized

    Returns: list of tokens
    """
    ## Tokenize
    doc = nlp(sentence)
    tokens = [token.text for token in doc]

    ## Convert token to id
    tokens = BERT_tokenizer.convert_tokens_to_ids(tokens)

    ## Truncating/Padding
    tokens = tokens[:configs['MAX_SEQ_LEN_W']]
    tokens.extend([BERT_tokenizer.pad_token_id
                   for _ in range(configs['MAX_SEQ_LEN_W'] - len(tokens))])

    return tokens


def sentence2chars(sentence: str) -> object:
    """
    Tokenized 'sentence' to characters, and convert characters
    to Unicode
    Args:
        sentence (str): sentence to be tokenized

    Returns: list of tokens

    """
    ## Tokenize and convert to ID (ID is order in Unicode)
    tokens = list(map(lambda x: ord(x), list(sentence)))
    
    ## Truncate/Padding
    tokens = tokens[:configs['MAX_SEQ_LEN_CH']]
    tokens.extend([0 for _ in range(configs['MAX_SEQ_LEN_CH'] - len(tokens))])

    return result


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


def concat_tensor(tensors: list) -> torch.tensor:
    """
    Concatenate vertically list of tensors together with
    reshape it
    Args:
        tensors (list): list of tensors to be stacked

    Returns: Single tensor

    """
    tensor = torch.vstack(tensors)
    return torch.transpose(tensor, 0, 1)


if __name__ == '__main__':
    sample = {
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "support_fact": [
            "Scott Derrickson",
            "Ed Wood"
        ],
        "context": "Adam Collis is an American filmmaker and actor. He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010. He also studied cinema at the University of Southern California from 1991 to 1997. Collis first work was the assistant director for the Scott Derrickson's short \"Love in the Ruins\" (1995). In 1998, he played \"Crankshaft\" in Eric Koyanagi's \"Hundred Percent\".",
        "score": 1
    }

    import json
    for result in get_entities_from_context(sample['context']):
        print(json.dumps(result, indent=2))
