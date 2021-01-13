import logging

from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn.functional as torch_f
import torch.optim as torch_optim
from tqdm import tqdm
import torch

from modules.utils import load_object, save_object
from modules.encoding.utils import concat_tensor
from configs import args

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().setLevel(logging.INFO)

BERT_PATH       = f"{args.init_path}/_pretrained/BERT/{args.bert_model}/"


class Para_Selector:
    """
    This class serves training, storing trained model, things related to training, inference
    """

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
        path_data_train = f"{args.init_path}/DFGN/backup_files/select_paras/dataset_train.pkl.gz"
        path_data_dev   = f"{args.init_path}/DFGN/backup_files/select_paras/dataset_dev.pkl.gz"
        path_data_test  = f"{args.init_path}/DFGN/backup_files/select_paras/dataset_test.pkl.gz"

        ## Create iterator for each dataset
        iterator_train  = self.get_iterator(path_data_train)
        iterator_dev    = self.get_iterator(path_data_dev)
        iterator_test   = self.get_iterator(path_data_test)
        logging.info("=> Successfully load data and create iterators")

        ## Prepare model
        model = BertForSequenceClassification.from_pretrained(BERT_PATH) \
            .to(args.device)
        optimizer = torch_optim.Adam(model.parameters())

        logging.info("=> Successfully create model")


        ##################################
        ## 2. Start training
        ##################################
        logging.info("2. Start training")

        best_accuracy = -10e10
        for nth_epoch in range(args.n_epochs):
            ### Traing and evaluate
            train_loss, train_accuracy  = self.train(model, iterator_train, optimizer)
            eval_loss, eval_accuracy    = self.evaluate(model, iterator_dev)

            logging.info("| n_epoch : {:3d} | train : loss {:6.3f} accu {:6.3f} | eval : loss {:6.3f} accu {:6.3f} |"
                         .format(nth_epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

            ### Store model if yield better result
            if best_accuracy < eval_accuracy:
                self.save_model(model)


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

        torch.save(model.state_dict(), "./backup_files/select_paras/paras_selector.pt")

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

    def train(self, model, iterator_train, optimizer):
        model.train()

        accum_loss, accum_accuracy = 0, 0


        for batch in tqdm(iterator_train):
            src             = concat_tensor(batch['sentence']).to(args.device)
            trg             = batch['score'].to(args.device)

            optimizer.zero_grad()

            outputs         = model(src, labels=trg, return_dict=True)

            loss            = outputs.loss
            accuracy        = self.cal_accuracy(outputs.logits, trg)


            loss.sum().backward()
            optimizer.step()


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
                trg             = batch['score'].to(args.device)

                outputs         = model(src, labels=trg, return_dict=True)

                loss            = outputs.loss
                accuracy        = self.cal_accuracy(outputs.logits, trg)


                accum_loss      += loss
                accum_accuracy  += accuracy

        return accum_loss / len(iterator_dev), accum_accuracy / len(iterator_dev)

    def save_model(self, model):
        """
        Save model during training
        :param model: Pytorch model
        """

        path_model = f"{args.init_path}/backup_files/select_paras/paras_select.pt"

        torch.save(model.state_dict(), path_model)

    def get_iterator(self, path):
        dataset = [
            {
                'sentence'  : data_point['sentence'],
                'score'     : data_point['score'],
            }
            for data_point in load_object(path)
        ]

        return DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


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
        logging.info("1. Prepare data and model")

        ## Load data
        path_data_train = f"{args.init_path}/DFGN/backup_files/select_paras/dataset_train.pkl.gz"
        path_data_dev   = f"{args.init_path}/DFGN/backup_files/select_paras/dataset_dev.pkl.gz"
        path_data_test  = f"{args.init_path}/DFGN/backup_files/select_paras/dataset_test.pkl.gz"

        ## Create iterator for each dataset
        iterator_train  = self.get_iterator(path_data_train)
        iterator_dev    = self.get_iterator(path_data_dev)
        iterator_test   = self.get_iterator(path_data_test)
        logging.info("=> Successfully load data and create iterators")


        ##################################
        ## 2. Load model
        ##################################
        model = BertForSequenceClassification.from_pretrained(BERT_PATH) \
            .to(args.device)
        model.load_state_dict(torch.load("./backup_files/select_paras/paras_selector.pt"))


        ##################################
        ## 3. Start inferring and store data
        ##################################
        selected_data_train = self.inference(model, iterator_train)
        selected_data_train.extend(self.inference(model, iterator_test))

        selected_data_dev   = self.inference(model, iterator_dev)

        save_object("./backup_files/select_paras/selected_data_train.pkl.gz",
                    selected_data_train)
        save_object("./backup_files/select_paras/selected_data_dev.pkl.gz",
                    selected_data_dev)

    def inference(self, model, iterator_inference):
        ## Following variable contains question and c
        results = list()

        model.eval()

        with torch.no_grad():
            for batch in iterator_inference:
                src = batch['sentence'].to(args.device)

                logits = model(src, labels=None, return_dict=True).logits

                ### Convert logits value to usable ones
                scores = torch_f.softmax(logits, dim=1)[:, 0]

                ### Save to 'results'

                for datapoint, score in zip(batch, scores):
                    results.append({
                        '_id'           : datapoint['_id'],
                        '_id_context'   : datapoint['_id_context'],
                        'score'         : score
                    })

        return results


if __name__ == '__main__':
    paras_seletor = Para_Selector()

    if args.task == "selectparas_train":
        paras_seletor.trigger_train()
    elif args.task == "selectparas_inference":
        paras_seletor.trigger_inference()
    else:
        logging.error("Err: Mismatch 'task' argument.")
