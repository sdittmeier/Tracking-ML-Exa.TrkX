"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""

# System imports
import sys
import os
import logging

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_cluster import radius_graph
import numpy as np

from .quantization_utils import quantize_features
import csv

# Local Imports
from .utils import graph_intersection, split_datasets, build_edges

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingBase(LightningModule):
    def __init__(self, hparams, bops_memory):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.bops_memory = bops_memory
        self.pruned = 0

    def setup(self, stage):
        self.trainset, self.valset, self.testset = split_datasets(**self.hparams)

    def train_dataloader(self):
        if len(self.trainset) > 0:
            return DataLoader(self.trainset, batch_size=1, num_workers=16)
        else:
            return None

    def val_dataloader(self):
        if len(self.valset) > 0:
            return DataLoader(self.valset, batch_size=1, num_workers=16)
        else:
            return None

    def test_dataloader(self):
        if len(self.testset):
            return DataLoader(self.testset, batch_size=1, num_workers=16)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def get_input_data(self, batch):

        fixed_point = self.hparams["input_quantization"]
        pre_point = self.hparams["integer_part"]
        post_point = self.hparams["fractional_part"]

        batch.x = quantize_features(batch.x.cpu(), False, fixed_point, pre_point, post_point).to('cuda:0')
        batch.cell_data = quantize_features(batch.cell_data.cpu(), False, fixed_point, pre_point, post_point).to('cuda:0')

        if self.hparams["cell_channels"] > 0:
            input_data = torch.cat(
                [batch.cell_data[:, : self.hparams["cell_channels"]], batch.x], axis=-1
            )
            input_data[input_data != input_data] = 0
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0

        return input_data

    def get_query_points(self, batch, spatial):

        if "query_all_points" in self.hparams["regime"]:
            query_indices = torch.arange(len(spatial)).to(spatial.device)
        elif "query_noise_points" in self.hparams["regime"]:
            query_indices = torch.cat(
                [torch.where(batch.pid == 0)[0], batch.signal_true_edges.unique()]
            )
        else:
            query_indices = batch.signal_true_edges.unique()

        query_indices = query_indices[torch.randperm(len(query_indices))][
            : self.hparams["points_per_batch"]
        ]
        query = spatial[query_indices]

        return query_indices, query

    def append_hnm_pairs(self, e_spatial, query, query_indices, spatial):

        if "low_purity" in self.hparams["regime"]:
            knn_edges = build_edges(
                query, spatial, query_indices, self.hparams["r_train"], 500
            )
            knn_edges = knn_edges[
                :,
                torch.randperm(knn_edges.shape[1])[
                    : int(self.hparams["r_train"] * len(query))
                ],
            ]

        else:
            knn_edges = build_edges(
                query,
                spatial,
                query_indices,
                self.hparams["r_train"],
                self.hparams["knn"],
            )

        e_spatial = torch.cat(
            [
                e_spatial,
                knn_edges,
            ],
            axis=-1,
        )

        return e_spatial

    def append_random_pairs(self, e_spatial, query_indices, spatial):
        n_random = int(self.hparams["randomisation"] * len(query_indices))
        indices_src = torch.randint(
            0, len(query_indices), (n_random,), device=self.device
        )
        indices_dest = torch.randint(0, len(spatial), (n_random,), device=self.device)
        random_pairs = torch.stack([query_indices[indices_src], indices_dest])

        e_spatial = torch.cat(
            [e_spatial, random_pairs],
            axis=-1,
        )
        return e_spatial

    def get_true_pairs(self, e_spatial, y_cluster, new_weights, e_bidir):
        e_spatial = torch.cat(
            [
                e_spatial.to(self.device),
                e_bidir,
            ],
            axis=-1,
        )
        y_cluster = torch.cat([y_cluster.int(), torch.ones(e_bidir.shape[1])])
        new_weights = torch.cat(
            [
                new_weights,
                torch.ones(e_bidir.shape[1], device=self.device)
                * self.hparams["weight"],
            ]
        )
        return e_spatial, y_cluster, new_weights

    def get_hinge_distance(self, spatial, e_spatial, y_cluster):

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        return hinge, d

    def get_truth(self, batch, e_spatial, e_bidir):

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        return e_spatial, y_cluster

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch)

        with torch.no_grad():
            spatial = self(input_data)

        query_indices, query = self.get_query_points(batch, spatial)

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            e_spatial = self.append_hnm_pairs(e_spatial, query, query_indices, spatial)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            e_spatial = self.append_random_pairs(e_spatial, query_indices, spatial)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], axis=-1
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)
        new_weights = y_cluster.to(self.device) * self.hparams["weight"]

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster, new_weights = self.get_true_pairs(
            e_spatial, y_cluster, new_weights, e_bidir
        )

        included_hits = e_spatial.unique()
        spatial[included_hits] = self(input_data[included_hits])

        hinge, d = self.get_hinge_distance(spatial, e_spatial, y_cluster)

        # Give negative examples a weight of 1 (note that there may still be TRUE examples that are weightless)
        new_weights[hinge == -1] = 1

        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        loss = negative_loss + self.hparams["weight"] * positive_loss
        if(self.hparams["l1_loss"]):
            l1_crit = torch.nn.L1Loss(size_average=False)
            reg_loss = 0
            for param in self.parameters():
                reg_loss += l1_crit(param,target=torch.zeros_like(param))

            factor = self.hparams["l1_factor"]
            loss += factor * reg_loss

        self.log("train_loss", loss, on_epoch=True, on_step=False)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False, verbose=False):

        input_data = self.get_input_data(batch)
        spatial = self(input_data)

        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], axis=-1
        )

        R95, R98, R99 = self.get_working_points(spatial, spatial, e_bidir)

        # Build whole KNN graph
        e_spatial_99 = build_edges(
            spatial, spatial, indices=None, r_max=R99, k_max=knn_num
        )

        e_spatial = build_edges(
            spatial, spatial, indices=None, r_max=knn_radius, k_max=knn_num
        )

        e_spatial_99, y_cluster_99 = self.get_truth(batch, e_spatial_99, e_bidir)

        _, d_99 = self.get_hinge_distance(
            spatial, e_spatial_99.to(self.device), y_cluster_99
        )
        pur_95, pur_98, pur_99 = self.get_working_metrics(e_spatial_99, y_cluster_99, d_99, R95, R98)

        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)

        hinge, d = self.get_hinge_distance(
            spatial, e_spatial.to(self.device), y_cluster
        )

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"]**2, reduction="mean"
        )

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        eff = cluster_true_positive / cluster_true
        pur = cluster_true_positive / cluster_positive

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "eff": eff, "pur": pur, "current_lr": current_lr, "R95": R95, "R98": R98, "R99": R99, "pur_95": pur_95, "pur_98": pur_98, "pur_99": pur_99, "total_bops": self.bops_memory["total_bops"], "total_mem_w_bits": self.bops_memory["total_mem_w_bits"], "total_mem_o_bits": self.bops_memory["total_mem_o_bits"], "pruned": self.pruned},
                on_epoch=True,
                on_step=False
            )

        if verbose:
            logging.info("Efficiency: {}".format(eff))
            logging.info("Purity: {}".format(pur))
            logging.info(batch.event_file)

        return {
            "loss": loss,
            "distances": d,
            "preds": e_spatial,
            "truth": y_cluster,
            "truth_graph": e_bidir,
            'eff': eff,
            'pur': pur
        }
    
    def get_working_points(self, spatial1, spatial2, truth):
        """
        Args:
            spatial (``torch.tensor``, required): The spatial embedding of the data
            truth (``torch.tensor``, required): The truth graph of the data

        Returns:
            ``torch.tensor``: The R95, R98, R99 values
        """
        # Get the R95, R98, R99 values
        # distances = torch.sum((spatial[truth[0]] - spatial[truth[1]])**2, dim=-1)
        distances = torch.pairwise_distance(spatial1[truth[0]], spatial2[truth[1]])
        # Sort the distances
        distances, indices = torch.sort(distances, descending=False)
        # Get the indices of the 95th, 98th, 99th percentile
        R95 = distances[int(len(distances)*0.95)]
        R98 = distances[int(len(distances)*0.98)]
        R99 = distances[int(len(distances)*0.99)]

        return R95.item(), R98.item(), R99.item()    

    def get_working_metrics(self, e_spatial, y_cluster, d, R95, R98):
        edge_mask_98 = d < R98**2
        e_spatial_98, y_cluster_98 = e_spatial[:, edge_mask_98], y_cluster[edge_mask_98]

        edge_mask_95 = d < R95**2
        e_spatial_95, y_cluster_95 = e_spatial[:, edge_mask_95], y_cluster[edge_mask_95]

        cluster_tp_99 = y_cluster.sum()
        cluster_tp_98 = y_cluster_98.sum()
        cluster_tp_95 = y_cluster_95.sum()

        cluster_positive_99 = len(e_spatial[0])
        cluster_positive_98 = len(e_spatial_98[0])
        cluster_positive_95 = len(e_spatial_95[0])
        
        pur_99 = cluster_tp_99 / cluster_positive_99
        pur_98 = cluster_tp_98 / cluster_positive_98
        pur_95 = cluster_tp_95 / cluster_positive_95

        return pur_95, pur_98, pur_99    

    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """

        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_val"], 150, log=True
        )

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_test"], 1000, log=False
        )

        return outputs

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """

        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()