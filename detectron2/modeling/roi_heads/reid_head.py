# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init



from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
import torch
from torch.autograd import Variable

import torchvision.models as models
from torch import nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck
import torchvision.models as models
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
import torch.nn.functional as F

import random
import cv2
import math

from .triplet_loss import _get_anchor_positive_triplet_mask, _get_anchor_negative_triplet_mask, _get_triplet_mask, _get_anchor_negative_triplet_mask_classes

ROI_REID_HEAD_REGISTRY = Registry("ROI_REID_HEAD")
ROI_REID_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_REID_HEAD_REGISTRY.register()
class REID_HEAD(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()
       
        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        #self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self._output_size = 128
        #self.bottle_neck = nn.Conv2d(256,128,3)
        #self.bottle_neck_2 = nn.Conv2d(128,64,3)
        self.fc = nn.Linear(7*7*256,512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.relu_fc = nn.ReLU(inplace=True)
        #self.fc_compare = nn.Linear(576,1)
        #weight_init.c2_xavier_fill(self.bottle_neck)
        #weight_init.c2_xavier_fill(self.bottle_neck_2)
        #weight_init.c2_xavier_fill(self.fc_compare)
        self.fc_out = nn.Linear(512,128)
        self.fc_compare = nn.Linear(128, 1)
        
        
        weight_init.c2_xavier_fill(self.fc)
        weight_init.c2_xavier_fill(self.fc_out)
        weight_init.c2_xavier_fill(self.fc_compare)

    def forward(self, x):
        
        
        #x = self.bottle_neck(x)
        #x = self.bn_fc(x)
        #x = self.relu_fc(x)
        #x = self.bottle_neck_2(x)
        #x = self.relu_fc(x)
        #x = x.view(x.size(0), -1)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_fc(x)
        x = self.relu_fc(x)
        x = self.fc_out(x)
        
        return x
    @property
    def output_size(self):
        return self._output_size
    def compare(self, e0, e1, train=False):
        out = torch.abs(e0 - e1)
        out = self.fc_compare(out)
        if not train:
            out = torch.sigmoid(out)
        return out
    
    
    def sum_losses(self,features,labels,classes, loss, margin, prec_at_k):
        """For Pretraining

        Function for preatrainind this CNN with the triplet loss. Takes a sample of N=PK images, P different
        persons, K images of each. K=4 is a normal parameter.

        [!] Batch all and batch hard should work fine. Take care with weighted triplet or cross entropy!!

        Args:
            batch (list): [images, labels], images are Tensor of size (N,H,W,C), H=224, W=112, labels Tensor of
            size (N)
        """
        
        
        features = Variable(features).cuda()
        
        labels = Variable(labels).cuda()
        #print(labels[0:10])
      
        
        
       
        embeddings = self.forward(features)
        if(embeddings.shape[0]!=labels.shape[0]):
            print(embeddings.shape)
            print(labels.shape)
            print("size mismatch!!!")
            print(embeddings.shape)
            print(labels.shape)
            
        if loss == "cross_entropy":
            m = _get_triplet_mask(labels).nonzero()
            
            e0 = []
            e1 = []
            e2 = []
            for p in m:
                e0.append(embeddings[p[0]])
                e1.append(embeddings[p[1]])
                e2.append(embeddings[p[2]])
            e0 = torch.stack(e0,0)
            e1 = torch.stack(e1,0)
            e2 = torch.stack(e2,0)

            out_pos = self.compare(e0, e1, train=True)
            out_neg = self.compare(e0, e2, train=True)

            tar_pos = Variable(torch.ones(out_pos.size(0)).view(-1,1).cuda())
            tar_neg = Variable(torch.zeros(out_pos.size(0)).view(-1,1).cuda())

            loss_pos = F.binary_cross_entropy_with_logits(out_pos, tar_pos)
            loss_neg = F.binary_cross_entropy_with_logits(out_neg, tar_neg)

            total_loss = (loss_pos + loss_neg)/2

        elif loss == 'batch_all':
            # works, batch all strategy
            m = _get_triplet_mask(labels).nonzero()
            e0 = []
            e1 = []
            e2 = []
            for p in m:
                e0.append(embeddings[p[0]])
                e1.append(embeddings[p[1]])
                e2.append(embeddings[p[2]])
            e0 = torch.stack(e0,0)
            e1 = torch.stack(e1,0)
            e2 = torch.stack(e2,0)
            total_loss = F.triplet_margin_loss(e0, e1, e2, margin=margin, p=2)
        elif loss == 'batch_hard':
            # compute pariwise square distance matrix, not stable with sqr as 0 can happen
            n = embeddings.size(0)
            m = embeddings.size(0)
            d = embeddings.size(1)
            
            x = embeddings.data.unsqueeze(1).expand(n, m, d)
            y = embeddings.data.unsqueeze(0).expand(n, m, d)
            
            dist = torch.pow(x - y, 2).sum(2)

            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            
            #print(mask_anchor_positive[0:10])
            mask_anchor_negative = _get_anchor_negative_triplet_mask_classes(labels,classes).float()
            
            #print(mask_anchor_negative[0:10])
            #print(mask_anchor_positive.shape)
            #print(dist.shape)
            pos_dist = dist * mask_anchor_positive
            # here add value so that not valid values can not be picked
            max_val = torch.max(dist)
            
            neg_dist = dist + max_val * (1.0 - mask_anchor_negative)
            
            #print(pos_dist)
            #print(neg_dist)
            # for each anchor compute hardest pair
            triplets = []
            for i in range(dist[0].size(0)):
                #print(torch.max(pos_dist[i],0)[1])
                pos = torch.max(pos_dist[i],0)[1].item()
                neg = torch.min(neg_dist[i],0)[1].item()
                triplets.append((i, pos, neg))
            

            e0 = []
            e1 = []
            e2 = []
            for p in triplets:
                e0.append(embeddings[p[0]])
                e1.append(embeddings[p[1]])
                e2.append(embeddings[p[2]])
            e0 = torch.stack(e0,0)
            e1 = torch.stack(e1,0)
            e2 = torch.stack(e2,0)
            total_loss = F.triplet_margin_loss(e0, e1, e2, margin=margin, p=2)

        elif loss == 'weighted_triplet':
            # compute pairwise distance matrix
            dist = []
            # iteratively construct the columns
            for e in embeddings:
                ee = torch.cat([e.view(1,-1) for _ in range(embeddings.size(0))],0)
                dist.append(F.pairwise_distance(embeddings, ee))
            dist = torch.cat(dist, 1)

            # First, we need to get a mask for every valid positive (they should have same label)
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
            pos_dist = dist * Variable(mask_anchor_positive.float())

            # Now every valid negative mask
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
            neg_dist = dist * Variable(mask_anchor_negative.float())

            # now get the weights for each anchor, detach because it should be a constant weighting factor
            pos_weights = Variable(torch.zeros(dist.size()).cuda())
            neg_weights = Variable(torch.zeros(dist.size()).cuda())
            for i in range(dist.size(0)):
                # make by line
                mask = torch.zeros(dist.size()).byte().cuda()
                mask[i] = 1
                pos_weights[mask_anchor_positive & mask] = F.softmax(pos_dist[mask_anchor_positive & mask], 0)
                neg_weights[mask_anchor_negative & mask] = F.softmin(neg_dist[mask_anchor_negative & mask], 0)
            pos_weights = pos_weights.detach()
            neg_weights = neg_weights.detach()
            pos_weight_dist = pos_dist * pos_weights
            neg_weight_dist = neg_dist * neg_weights

            triplet_loss = torch.clamp(margin + pos_weight_dist.sum(1, keepdim=True) - neg_weight_dist.sum(1, keepdim=True), min=0)
            total_loss = triplet_loss.mean()
        else:
            raise NotImplementedError("Loss: {}".format(loss))

        losses = {}

        if prec_at_k:
            # compute pariwise square distance matrix, not stable with sqr as 0 can happen
            n = embeddings.size(0)
            m = embeddings.size(0)
            d = embeddings.size(1)

            x = embeddings.data.unsqueeze(1).expand(n, m, d)
            y = embeddings.data.unsqueeze(0).expand(n, m, d)

            dist = torch.pow(x - y, 2).sum(2)
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
            _, indices = torch.sort(dist, dim=1)
            num_hit = 0
            num_ges = 0
            for i in range(dist.size(0)):
                d = mask_anchor_positive[i].nonzero().view(-1,1)
                ind = indices[i][:prec_at_k+1]

                same = d==ind
                num_hit += same.sum()
                num_ges += prec_at_k
            k_loss = torch.Tensor(1)
            k_loss[0] = num_hit / num_ges
            losses['prec_at_k'] = Variable(k_loss.cuda())
        print(losses['prec_at_k'])
        losses['reid_loss'] = total_loss*0.2

        return losses

def build_reid_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = "ROI_REID_HEAD"
    
    return REID_HEAD(cfg, input_shape)
