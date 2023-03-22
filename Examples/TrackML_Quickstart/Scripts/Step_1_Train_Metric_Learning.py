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
from qonnx.util.inference_cost import inference_cost
from brevitas.quant_tensor import QuantTensor
import numpy as np

import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
import onnx.shape_inference as si
import torch.nn.utils.prune as prune

import copy

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("1_Train_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()

last_pruned = 0
val_loss = []

# from https://github.com/fastmachinelearning/qonnx/blob/main/tests/transformation/test_channelslast.py#L75
def get_golden_in_and_output(onnx_file, input_tensor): #, input_shape):
    #input_tensor = input_tensor.cpu().detach().numpy().astype(np.float32)
#    input_tensor = input_tensor.reshape(input_shape)
    model = ModelWrapper(onnx_file)
    model = ModelWrapper(si.infer_shapes(model.model))
    input_dict = {model.graph.input[0].name: input_tensor}
    golden_output_dict = oxe.execute_onnx(model, input_dict)
    golden_result = golden_output_dict[model.graph.output[0].name]

    return golden_result

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
        if(not metric_learning_configs["pruning_allow"]):
            return False
        global last_pruned
        global val_loss        
        export_path = f"pruning_{epoch}.onnx"
        export_json = f"pruning_{epoch}.json"
        val_loss.append(trainer.callback_metrics['val_loss'].cpu().numpy())  # could include feedback from validation or training loss here
        if(len(val_loss) > 10):
            val_loss.pop(0)
            if( (max(val_loss) - min(val_loss)) < pruning_val_loss):
                last_pruned = epoch
                logging.info(headline("Val_loss: Pruning" ))
                val_loss=[]
                model_copy = copy.deepcopy(model)
                parameters_to_prune_copy = [(model_copy.network[1], "weight"), (model_copy.network[4], "weight"), (model_copy.network[7], "weight"), (model_copy.network[10], "weight"), (model_copy.network[13], "weight")]
                if(last_pruned > 0):
                    for paras in parameters_to_prune_copy:
                        prune.remove(paras[0], name = 'weight')
                BrevitasONNXManager.export(model_copy, export_path = export_path, input_t = input_quant_tensor) # exporting the model to calculate BOPs just before we do pruning
                del model_copy
                inference_cost(export_path, output_json = export_json, discount_sparsity = True)
                return True
        if(((epoch-last_pruned) % pruning_freq)==(pruning_freq-1)):
            last_pruned = epoch
            logging.info(headline("Epoch: Pruning" ))
            val_loss=[]
            model_copy = copy.deepcopy(model)
            parameters_to_prune_copy = [(model_copy.network[1], "weight"), (model_copy.network[4], "weight"), (model_copy.network[7], "weight"), (model_copy.network[10], "weight"), (model_copy.network[13], "weight")]
            if(last_pruned > 0):
                for paras in parameters_to_prune_copy:
                    prune.remove(paras[0], name = 'weight')
            BrevitasONNXManager.export(model_copy, export_path = export_path, input_t = input_quant_tensor) # exporting the model to calculate BOPs just before we do pruning
            print(model_copy.network[4].weight)
            print(model.network[4].weight)
            del model_copy
            inference_cost(export_path, output_json = export_json, discount_sparsity = True)
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
                pruning_fn = metric_learning_configs["pruning_fn"],
                parameters_to_prune = parameters_to_prune,
                amount = metric_learning_configs["pruning_amount"],
                apply_pruning = apply_pruning,
                verbose = 1 #2 for per-layer sparsity, #1 for overall sparsity
            )
        ]
    )

    # adapt for quantization
    model.setup(stage="fit") # This is needed for the model to build its dataset(s)

    fixed_point = metric_learning_configs["input_quantization"]
    pre_point = metric_learning_configs["integer_part"]
    post_point = metric_learning_configs["fractional_part"]

    ev_size = 0
    logging.info("quantizing trainset")
    for event in model.trainset:
        if(event.x.size(dim=0) > ev_size):
            max_event = event
        ev_size=max(ev_size,event.x.size(dim=0))
#        print(event.x.size())
#        print(event.cell_data.size())
        event.x = quantize_features(event.x, False, fixed_point, pre_point, post_point)
        event.cell_data = quantize_features(event.cell_data, False, fixed_point, pre_point, post_point)

    logging.info("quantizing valset")
    for event in model.valset:        
        if(event.x.size(dim=0) > ev_size):
            max_event = event
        ev_size=max(ev_size,event.x.size(dim=0))
        event.x = quantize_features(event.x, False, fixed_point, pre_point, post_point)
        event.cell_data = quantize_features(event.cell_data, False, fixed_point, pre_point, post_point)
    
    logging.info("quantizing testset")
    for event in model.testset:
        if(event.x.size(dim=0) > ev_size):
            max_event = event
        ev_size=max(ev_size,event.x.size(dim=0))
        event.x = quantize_features(event.x, False, fixed_point, pre_point, post_point)
        event.cell_data = quantize_features(event.cell_data, False, fixed_point, pre_point, post_point)

#    print(ev_size)

#    input_shape = (ev_size, 12) # (batchsize can always be 1, channel (node features) = 3 + 9, ev_size for maximum eventsize)
    input_tensor = torch.cat(
                [max_event.cell_data[:, : metric_learning_configs["cell_channels"]], max_event.x], axis=-1
            )
#    print(input_tensor)
#    print(input_tensor.size())
    input_bitwidth = pre_point + post_point + 1 # for sign +1 !
    scale_array = np.full((1,12),1./(input_bitwidth)) #12 features
    scale_tensor = torch.from_numpy(scale_array)
    zp = torch.tensor(0.0)
    signed = True
    input_quant_tensor = QuantTensor(input_tensor.to('cuda:0'), scale_tensor, zp, input_bitwidth, signed, training = False)

    trainer.fit(model)

    logging.info(headline("c) Saving model") )

    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))
    
    export_onnx_path = "pruning_final.onnx"
    input_quant_tensor = QuantTensor(input_tensor, scale_tensor, zp, input_bitwidth, signed, training = False)
    BrevitasONNXManager.export(model, export_path = export_onnx_path, input_t = input_quant_tensor)

#    input_shape = (ev_size,12)
#    get_golden_in_and_output(export_onnx_path, input_quant_tensor)#, input_shape)

#    wandb.finish()
    return trainer, model


if __name__ == "__main__":

#    args = parse_args()
#    config_file = args.config

    trainer, model = train()    

