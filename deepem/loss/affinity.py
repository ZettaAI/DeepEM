import numpy as np

import torch
import torch.nn as nn

from deepem.utils import torch_utils


class EdgeSampler(object):
    def __init__(self, edges, split_boundary=True):
        self.edges = list(edges)
        self.split_boundary = split_boundary

    def generate_edges(self):
        return list(self.edges)

    def generate_true_aff(self, obj, edge):
        o1, o2 = torch_utils.get_pair(obj, edge)
        if self.split_boundary:
            ret = ((o1 == o2) & (o1 != 0) & (o2 != 0))
        else:
            ret = (o1 == o2)
        return ret.type(obj.type())

    def generate_mask_aff(self, mask, edge):
        m1, m2 = torch_utils.get_pair(mask, edge)
        return (m1 * m2).type(mask.type())


class EdgeCRF(nn.Module):
    def __init__(self, criterion, size_average=False, class_balancer=None):
        super(EdgeCRF, self).__init__()
        self.criterion = criterion
        self.size_average = size_average
        self.class_balancer = class_balancer

    def forward(self, preds, targets, masks, edges=None):
        """
        Args:
            preds: List of predictions per edge
            targets: List of targets per edge
            masks: List of masks per edge
            edges: Optional list of edge tuples for channel mapping
        """
        assert len(preds) == len(targets) == len(masks)
        loss, nmsk = 0, 0

        for i, (pred, target, mask) in enumerate(zip(preds, targets, masks)):
            # Apply directional class balancing if edges provided
            if self.class_balancer is not None and edges is not None:
                channel = self._edge_to_channel(edges[i])
                mask = self.class_balancer(target, mask, channel=channel)
            elif self.class_balancer is not None:
                # Backward compatibility: no edges provided
                mask = self.class_balancer(target, mask)

            l, n = self.criterion(pred, target, mask)
            loss += l
            nmsk += n

        assert nmsk.item() >= 0

        if nmsk.item() == 0:
            loss = torch.tensor(0).type(torch.cuda.FloatTensor)
            return loss, nmsk

        if self.size_average:
            assert nmsk.item() > 0
            try:
                loss = loss / nmsk.item()
                nmsk = torch.tensor(1, dtype=nmsk.dtype, device=nmsk.device)
            except:
                import pdb; pdb.set_trace()
                raise

        return loss, nmsk

    def _edge_to_channel(self, edge):
        """Map edge tuple to channel index for weight ordering.

        Args:
            edge: Edge tuple in (z, y, x) format

        Returns:
            Channel index: 0 for x, 1 for y, 2 for z
        """
        # Find which dimension the edge spans
        edge_tuple = tuple(edge)
        if edge_tuple[-1] != 0:  # x-dimension
            return 0
        elif edge_tuple[-2] != 0:  # y-dimension
            return 1
        elif edge_tuple[-3] != 0:  # z-dimension
            return 2
        else:
            # Shouldn't happen with valid edges
            return 0


class AffinityLoss(nn.Module):
    def __init__(self, edges, criterion, split_boundary=True,
                 size_average=False, class_balancer=None):
        super(AffinityLoss, self).__init__()
        self.sampler = EdgeSampler(edges, split_boundary=split_boundary)
        self.decoder = AffinityLoss.Decoder(edges)
        self.edges = edges  # Store edges for passing to EdgeCRF
        self.criterion = EdgeCRF(
            criterion,
            size_average=size_average,
            class_balancer=class_balancer
        )

    def forward(self, preds, label, mask):
        pred_affs = list()
        true_affs = list()
        mask_affs = list()
        edges = self.sampler.generate_edges()
        for i, edge in enumerate(edges):
            try:
                pred_affs.append(self.decoder(preds, i))
                true_affs.append(self.sampler.generate_true_aff(label, edge))
                mask_affs.append(self.sampler.generate_mask_aff(mask, edge))
            except:
                raise
        # Pass edges to criterion for channel mapping
        return self.criterion(pred_affs, true_affs, mask_affs, edges=edges)

    class Decoder(nn.Module):
        def __init__(self, edges):
            super(AffinityLoss.Decoder, self).__init__()
            assert len(edges) > 0
            self.edges = list(edges)

        def forward(self, x, i):
            num_channels = x.size(-4)
            assert num_channels == len(self.edges)
            assert i < num_channels and i >= 0
            edge = self.edges[i]
            return torch_utils.get_pair_first(x[...,[i],:,:,:], edge)
