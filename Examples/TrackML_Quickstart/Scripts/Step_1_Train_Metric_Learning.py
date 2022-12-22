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

    threshold = 0 # relative threshold, to account for different max values per feature
    quantizers = learn_quantization(model.trainset, threshold)
    print(quantizers)
    
    print("quantizing trainset")
    for event in model.trainset:
#        print(event)
#        print(event.x)
#        print(event.cell_data)
        event.x = quantize_features(event.x, quantizers[:3])
        event.cell_data = quantize_features(event.cell_data, quantizers[3:])
#        print(event)
#        print(event.x)
#        print(event.cell_data)
    
    print("quantizing valset")
    for event in model.valset:
        event.x = quantize_features(event.x, quantizers[:3])
        event.cell_data = quantize_features(event.cell_data, quantizers[3:])
    
    print("quantizing testset")
    for event in model.testset:
        event.x = quantize_features(event.x, quantizers[:3])
        event.cell_data = quantize_features(event.cell_data, quantizers[3:])


    trainer.fit(model)

    logging.info(headline("c) Saving model") )

    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    trainer, model = train(config_file)    

