import logging
import os

from transformers import BertForSequenceClassification, AdamW,\
                         get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn.functional as torch_f
from tqdm import tqdm
import torch

from modules.para_selection.DataHelper import CustomizedDataset
from modules.utils import load_object, save_object, check_file_existence
from modules.encoding.utils import concat_tensor
from configs import args

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().setLevel(logging.INFO)

BERT_PATH       = f"{args.init_path}/_pretrained/BERT/{args.bert_model}/"


class Para_Selector:
    """
    This class serves training, storing trained model, things related to training, inference
    """

    def __init__(self):
        self.path_saved_model    = f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/paras_select.pt"

        self.path_data_train     = f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/dataset_train.pkl.gz"
        self.path_data_dev       = f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/dataset_dev.pkl.gz"
        self.path_data_test      = f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/dataset_test.pkl.gz"

        self.path_inferred_train = f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/score_data_train.pkl.gz"
        self.path_inferred_dev   = f"{args.init_path}/_data/QA/HotpotQA/backup_files/select_paras/score_data_dev.pkl.gz"

    ##################################################
    # Methods serving training purpose
    ##################################################
    def trigger_train(self):
        """
        Training model paragraph selection
        """

        ##################################
        ## 1. Prepare data, model
        ##################################
        logging.info("1. Prepare data and model")

        ## Load data


        ## Create iterator for each dataset
        iterator_train  = self.get_iterator(self.path_data_train)
        iterator_dev    = self.get_iterator(self.path_data_dev)
        iterator_test   = self.get_iterator(self.path_data_test)
        logging.info("=> Successfully load data and create iterators")

        ## Prepare model
        if torch.cuda.device_count() > 1:
            try:
                model           = torch.nn.DataParallel(BertForSequenceClassification.from_pretrained(BERT_PATH))\
                    .to(args.device)
            except OSError:
                model = torch.nn.DataParallel(BertForSequenceClassification.from_pretrained(args.bert_model)) \
                    .to(args.device)
        else:
            try:
                model = BertForSequenceClassification.from_pretrained(BERT_PATH).to(args.device)
            except OSError:
                model = BertForSequenceClassification.from_pretrained(args.bert_model).to(args.device)

        ## Check if model's save instance is available, load it
        model           = self.load_model(model)

        optimizer       = AdamW(model.parameters(),
                                lr=2e-5, eps=1e-8)
        total_steps     = len(iterator_train) * args.n_epochs
        scheduler       = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                          num_training_steps=total_steps)

        logging.info("=> Successfully create model")


        ##################################
        ## 2. Start training
        ##################################
        logging.info("2. Start training")

        best_accuracy = -10e10
        for nth_epoch in range(args.n_epochs):
            ### Train and evaluate
            logging.info(f"== Epoch: {nth_epoch+1:3d}")
            logging.info(f"=== Training")
            train_loss, train_accuracy  = self.train(model, iterator_train, optimizer, scheduler)
            logging.info(f"=== Evaluating")
            eval_loss, eval_accuracy    = self.evaluate(model, iterator_dev)


            logging.info("| n_epoch : {:3d} | train : loss {:6.3f} accu {:6.3f} | eval : loss {:6.3f} accu {:6.3f} |"
                         .format(nth_epoch+1, train_loss, train_accuracy, eval_loss, eval_accuracy))

            ### Store model if yield better result
            if best_accuracy < eval_accuracy:
                best_accuracy = eval_accuracy
                self.save_model(model)

                logging.info("Successfully save model")


        logging.info("=> Training finishes.")


        ##################################
        ## 3. Start testing with test data
        ##################################
        logging.info("3. Test with test dataset")

        test_loss, test_accuracy = self.evaluate(model, iterator_test)
        logging.info("=> Test result")
        logging.info("| test : loss {:6.3f} accu {:6.3f} |".format(test_loss, test_accuracy))

        ##################################
        ## 4. Store model
        ##################################
        logging.info("4. Store model")

        torch.save(model.state_dict(), self.path_saved_model)

    def cal_accuracy(self, logist, target):
        """
        Calculate accuracy during training and evaluating
        :param logist: tensor containing predicted results
        :type logist: torch.tensor
        :param target: tensor containing predicted results
        :type target: torch.tensor
        :return: accuracy
        :rtype: float
        """

        selections = torch.tensor([
            1 if k[1] >= 0.5 else 0
            for k in torch_f.softmax(logist, dim=1)]).to(args.device)

        correct         = (selections == target).float()

        return correct.sum() / len(correct)

    def train(self, model, iterator_train, optimizer, scheduler):
        model.train()

        accum_loss, accum_accuracy = 0, 0
        for batch in tqdm(iterator_train):
            src             = concat_tensor(batch['sentence']).to(args.device)
            input_masks     = concat_tensor(batch['attn_mask']).to(args.device)
            trg             = batch['score'].to(args.device)

            optimizer.zero_grad()

            outputs         = model(src, labels=trg, token_type_ids=None,
                                    attention_mask=input_masks, return_dict=True)

            loss            = outputs.loss.mean()
            accuracy        = self.cal_accuracy(outputs.logits, trg)

            loss.backward()
            optimizer.step()
            scheduler.step()

            accum_loss      += loss
            accum_accuracy  += accuracy

        return accum_loss / len(iterator_train), accum_accuracy / len(iterator_train)

    def evaluate(self, model, iterator_dev):
        """
        Evaluate model during training
        :param model: Pytorch model
        :param iterator_dev: iterator of dev dataset
        """

        model.eval()

        accum_loss, accum_accuracy = 0, 0
        with torch.no_grad():
            for batch in tqdm(iterator_dev):
                src             = concat_tensor(batch['sentence']).to(args.device)
                input_masks     = concat_tensor(batch['attn_mask']).to(args.device)
                trg             = batch['score'].to(args.device)

                outputs         = model(src, labels=trg, token_type_ids=None,
                                        attention_mask=input_masks, return_dict=True)

                loss            = outputs.loss.mean()
                accuracy        = self.cal_accuracy(outputs.logits, trg)


                accum_loss      += loss
                accum_accuracy  += accuracy

        return accum_loss / len(iterator_dev), accum_accuracy / len(iterator_dev)

    def save_model(self, model):
        """
        Save model during training
        :param model: Pytorch model
        """

        ## Check if folder existed, otherwise create
        try:
            os.makedirs(os.path.dirname(self.path_saved_model))
        except FileExistsError:
            ## If reach here, folder(s) is created
            pass

        torch.save(model.state_dict(), self.path_saved_model)

    def get_iterator(self, path):
        dataset = [
            {
                'sentence'  : data_point['sentence'],
                'attn_mask' : data_point['attn_mask'],
                'score'     : data_point['score'],
            }
            for data_point in load_object(path)
        ]

        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_cpus)


    ##################################################
    # Methods serving inferring purpose
    ##################################################
    def trigger_inference(self):
        """
        Trigger inferring process
        """
        ##################################
        ## 1. Read 3 files 'dataset_.pkl.gz' and
        ## reconstruct DataLoader from those
        ##################################
        logging.info("1. Prepare data")


        ## Load data and create iterator for each dataset
        iterator_train  = self.get_iterator(self.path_data_train)
        iterator_dev    = self.get_iterator(self.path_data_dev)
        iterator_test   = self.get_iterator(self.path_data_test)
        logging.info("=> Successfully load data and create iterators")


        ##################################
        ## 2. Load model
        ##################################
        logging.info("2. Prepare model")
        try:
            model = torch.nn.DataParallel(BertForSequenceClassification.from_pretrained(BERT_PATH)) \
                .to(args.device)
        except OSError:
            model = torch.nn.DataParallel(BertForSequenceClassification.from_pretrained(args.bert_model)) \
                .to(args.device)

        model = self.load_model(model)


        ##################################
        ## 3. Start inferring and store data
        ##################################
        logging.info("3. Start inferring")

        logging.info("=> Infer training data")
        inferred_data_train = self.inference(model, iterator_train)
        inferred_data_train.extend(self.inference(model, iterator_test))

        logging.info("=> Infer dev data")
        inferred_data_dev   = self.inference(model, iterator_dev)


        save_object(self.path_inferred_train, inferred_data_train)
        save_object(self.path_inferred_dev, inferred_data_dev)
        logging.info("Successfully save inferred data.")


    def inference(self, model, iterator_inference):
        ## Following variable contains question and c
        results = list()

        model.eval()

        with torch.no_grad():
            for batch in tqdm(iterator_inference):
                src         = concat_tensor(batch['sentence']).to(args.device)
                input_masks = concat_tensor(batch['attn_mask']).to(args.device)
                trg         = batch['score'].to(args.device)

                outputs     = model(src, labels=trg, token_type_ids=None,
                                    attention_mask=input_masks, return_dict=True)

                logits      = outputs.logits

                ### Convert logits value to usable ones
                scores      = torch_f.softmax(logits, dim=1)[:, 0]

                ### Save to 'results'
                for datapoint, score in zip(batch, scores):
                    results.append({
                        '_id'           : datapoint['_id'],
                        '_id_context'   : datapoint['_id_context'],
                        'score'         : score
                    })

        return results

    def load_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Load model from path

        Args:
            model (object): model to be load from saved instance if existed

        Returns:
            model even loaded or not
        """

        if check_file_existence(self.path_saved_model):
            logging.info("=> Saved model instance existed. Load it.")
            model.load_state_dict(torch.load(self.path_saved_model))

        return model


if __name__ == '__main__':
    paras_seletor = Para_Selector()

    if args.task == "selectparas_train":
        paras_seletor.trigger_train()
    elif args.task == "selectparas_inference":
        paras_seletor.trigger_inference()
    else:
        logging.error("Err: Mismatch 'task' argument.")
