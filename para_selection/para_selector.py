import logging

from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn.functional as torch_f
import torch.optim as torch_optim
import torch

from utils import load_object
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
        path_data_train = f"{args.init_path}/DFGN-hoangle/backup_files/select_paras/dataset_train.pkl.gz"
        path_data_dev   = f"{args.init_path}/DFGN-hoangle/backup_files/select_paras/dataset_dev.pkl.gz"
        path_data_test  = f"{args.init_path}/DFGN-hoangle/backup_files/select_paras/dataset_test.pkl.gz"

        ## Create iterator for each dataset
        # iterator_train  = CustomizedDataLoader(path_data_train, args.batch_size, args.n_cpus)
        # iterator_dev    = CustomizedDataLoader(path_data_dev, args.batch_size, args.n_cpus)
        # iterator_test   = CustomizedDataLoader(path_data_test, args.batch_size, args.n_cpus)

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
        logging.info("4. Test with test dataset")


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

        for batch in iterator_train:
            src             = batch['sentence'].to(args.device)
            trg             = batch['score'].to(args.device)

            optimizer.zero_grad()

            outputs         = model(src, labels=trg, return_dict=True)

            loss            = outputs.loss
            accuracy        = self.cal_accuracy(outputs.logits, trg)


            loss.backward()
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
            for batch in iterator_dev:
                src             = batch['sentence'].to(args.device)
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
                'sentence': torch.from_numpy(data_point['sentence']),
                'score': torch.from_numpy(data_point['score']),
            }
            for data_point in load_object(path)
        ]

        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_cpus)

    ##################################################
    # Methods serving training purpose
    ##################################################
    def inference(self, model):
        pass


if __name__ == '__main__':
    paras_seletor = Para_Selector()

    paras_seletor.trigger_train()
