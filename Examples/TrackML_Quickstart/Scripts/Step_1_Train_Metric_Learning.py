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
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
import torch

sys.path.append("../../")

from Pipelines.TrackML_Example.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from utils.convenience_utils import headline
sys.path.append("../../")
from Pipelines.TrackML_Example.LightningModules.Embedding.quantization_utils import quantize_features
import csv

import wandb

from brevitas.export.onnx.generic.manager import BrevitasONNXManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("1_Train_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()

last_pruned = 0
val_loss = []

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
    # this should actually work
    # metric_learning_configs.update(dict(wandb.config))
    metric_learning_configs = dict(wandb.config)

    logging.info(headline("a) Initialising model"))

    model = LayerlessEmbedding(metric_learning_configs)

    logging.info(headline("b) Running training" ))

    save_directory = os.path.join(common_configs["artifact_directory"], "metric_learning")
    logger = WandbLogger(save_directory)#, project=common_configs["experiment_name"])

    # placeholders
    parameters_to_prune = [(model.network[1], "weight"), (model.network[4], "weight"), (model.network[7], "weight"), (model.network[10], "weight"), (model.network[13], "weight")]
    pruning_freq = metric_learning_configs["pruning_freq"]
    pruning_val_loss = metric_learning_configs["pruning_val_loss"]

    def apply_pruning(epoch):
        global last_pruned
        global val_loss
        val_loss.append(trainer.callback_metrics['val_loss'].cpu().numpy())  # could include feedback from validation or training loss here
#        logging.info(val_loss)
        if(len(val_loss) > 10):
            val_loss.pop(0)
#            logging.info(max(val_loss))
#            logging.info(min(val_loss))
            if( (max(val_loss) - min(val_loss)) < pruning_val_loss):
                last_pruned = epoch
                logging.info(headline("Val_loss: Pruning" ))
                val_loss=[]
                return True
        if(((epoch-last_pruned) % pruning_freq)==(pruning_freq-1)):
            last_pruned = epoch
            logging.info(headline("Epoch: Pruning" ))
            val_loss=[]
            return True
        else:
            return False

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else None,
        gpus=common_configs["gpus"],
        max_epochs=metric_learning_configs["max_epochs"],
        logger=logger,
        callbacks=[
            ModelPruning(
                pruning_fn="l1_unstructured",
                parameters_to_prune= parameters_to_prune,
                amount = metric_learning_configs["pruning_amount"],
                apply_pruning = apply_pruning,
                verbose = 2
            )
        ]
    )

    # adapt for quantization
    model.setup(stage="fit") # This is needed for the model to build its dataset(s)

    threshold = 1e-6 # relative threshold, to account for different max values per feature
    fixed_point = metric_learning_configs["input_quantization"]
    pre_point = metric_learning_configs["integer_part"]
    post_point = metric_learning_configs["fractional_part"]

    ev_size = 0
    logging.info("quantizing trainset")
    for event in model.trainset:
        ev_size=max(ev_size,event.x.size(dim=0))
        event.x = quantize_features(event.x, False, fixed_point, pre_point, post_point)
        event.cell_data = quantize_features(event.cell_data, False, fixed_point, pre_point, post_point)

    logging.info("quantizing valset")
    for event in model.valset:
        ev_size=max(ev_size,event.x.size(dim=0))
        event.x = quantize_features(event.x, False, fixed_point, pre_point, post_point)
        event.cell_data = quantize_features(event.cell_data, False, fixed_point, pre_point, post_point)
    
    logging.info("quantizing testset")
    for event in model.testset:
        ev_size=max(ev_size,event.x.size(dim=0))
        event.x = quantize_features(event.x, False, fixed_point, pre_point, post_point)
        event.cell_data = quantize_features(event.cell_data, False, fixed_point, pre_point, post_point)

    print(ev_size)

    input_shape = (1, 12, 512)#ev_size) # (batchsize can always be 1, channel (node features) = 3 + 9, ev_size for maximum eventsize)
    export_onnx_path = "test_brevitas_onnx.onnx"
    BrevitasONNXManager.export(model, input_shape, export_onnx_path)

    trainer.fit(model)

    logging.info(headline("c) Saving model") )

    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))



    wandb.finish()
    return trainer, model


if __name__ == "__main__":

#    args = parse_args()
#    config_file = args.config

    trainer, model = train()    

