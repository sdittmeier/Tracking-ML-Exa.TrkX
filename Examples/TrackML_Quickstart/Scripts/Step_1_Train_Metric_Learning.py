"""
This script runs step 1 of the TrackML Quickstart example: Training the metric learning model.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import torch

sys.path.append("../../")

from Pipelines.TrackML_Example.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from utils.convenience_utils import headline
from utils.quantization_utils import learn_quantization, quantize_features
import csv

import wandb

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("1_Train_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def train(config_file="pipeline_config.yaml"):

    logging.info(headline("Step 1: Running metric learning training"))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    metric_learning_configs = all_configs["metric_learning_configs"]

    wandb.init(
        # set the wandb project where this run will be logged
        project="qat_trackml_seb",
        
        # track hyperparameters and run metadata
        config=metric_learning_configs
    )

    logging.info(headline("a) Initialising model"))

    model = LayerlessEmbedding(metric_learning_configs)

    logging.info(headline("b) Running training" ))

    save_directory = os.path.join(common_configs["artifact_directory"], "metric_learning")
    logger = CSVLogger(save_directory, name=common_configs["experiment_name"])

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else None,
        gpus=common_configs["gpus"],
        max_epochs=metric_learning_configs["max_epochs"],
        logger=logger
    )

    # adapt for quantization
    model.setup(stage="fit") # This is needed for the model to build its dataset(s)

    threshold = 1e-6 # relative threshold, to account for different max values per feature
    fixed_point = True
    pre_point = 5
    post_point = 20
#    quantizers = learn_quantization(model.trainset, threshold)
    with open('testquantization.txt', 'r') as f:
        reader = csv.reader(f)
        quantizers = list(reader)
    for x in range(len(quantizers)):
        quantizers[x][0] = int(quantizers[x][0])
        quantizers[x][1] = float(quantizers[x][1])
        quantizers[x][2] = (quantizers[x][2] == ' True')
    print(quantizers)
    
    print("quantizing trainset")
    for event in model.trainset:
        event.x = quantize_features(event.x, quantizers[:3], False, fixed_point, pre_point, post_point)
        event.cell_data = quantize_features(event.cell_data, quantizers[3:], False, fixed_point, pre_point, post_point)
    
    print("quantizing valset")
    for event in model.valset:
        event.x = quantize_features(event.x, quantizers[:3], True, fixed_point, pre_point, post_point)
        event.cell_data = quantize_features(event.cell_data, quantizers[3:], True, fixed_point, pre_point, post_point)
    
    print("quantizing testset")
    for event in model.testset:
        event.x = quantize_features(event.x, quantizers[:3], False, fixed_point, pre_point, post_point)
        event.cell_data = quantize_features(event.cell_data, quantizers[3:], False, fixed_point, pre_point, post_point)


    trainer.fit(model)

    logging.info(headline("c) Saving model") )

    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))

    wandb.finish()
    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    trainer, model = train(config_file)    

