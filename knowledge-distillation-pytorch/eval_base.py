"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.resnet as resnet
import model.data_loader as data_loader
import torch.optim as optim
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory of params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--device', default='gpu', help="CPU or GPU")
parser.add_argument('--dataset', default='cifar10', help="Dataset to use")

def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        # compute model output
        start = time.perf_counter()
        output_batch = model(data_batch)
        torch.cuda.synchronize()
        end=time.perf_counter()
        print("Time(ms) : {:.04}".format((end-start)*1000))
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
     """
         Evaluate the model on a dataset for one pass.
     """
     # Load the parameters
     args = parser.parse_args()
     json_path = os.path.join(args.model_dir, 'params.json')
     assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
     params = utils.Params(json_path)

     # use GPU if available
     params.cuda = torch.cuda.is_available() and args.device == "gpu"    # use GPU is available
     params.dataset = args.dataset
     
     # Set the random seed for reproducible experiments
     torch.manual_seed(230)
     if params.cuda: torch.cuda.manual_seed(230)
        
     # Get the logger
     utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

     # Create the input data pipeline
     logging.info("Loading the dataset...")

     # fetch dataloaders
     # train_dl = data_loader.fetch_dataloader('train', params)
     dev_dl = data_loader.fetch_dataloader('dev', params)

     logging.info("- done.")

     # Define the model graph
     if params.model_version == "resnet18" or params.model_version == "res50-res18_distill":
         model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
     elif params.model_version == "resnet34" or params.model_version == "res50-res34_distill":
         model = resnet.ResNet34().cuda() if params.cuda else resnet.ResNet34()
     elif params.model_version == "resnet50":
         model = resnet.ResNet50().cuda() if params.cuda else resnet.ResNet50()
         model = torch.nn.DataParallel(model)
     elif params.model_version == "res50-cnn_distill":
         model = net.Net(params).cuda() if params.cuda else net.Net(params)
     # fetch loss function and metrics
     loss_fn = net.loss_fn
     metrics = resnet.metrics
    
     logging.info("Starting evaluation...")

     # Reload weights from the saved file
     utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)#, checkpointName='net')

     # Evaluate
     test_metrics = evaluate(model, loss_fn, dev_dl, metrics, params)
     save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
     utils.save_dict_to_json(test_metrics, save_path)
