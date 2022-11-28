# -*- coding: utf-8 -*-
from __future__ import division

import logging
import numpy as np
import torch
import torch.nn as nn
from torch import sigmoid
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.functional import linear, relu, softmax, logsigmoid, tanh, cosine_similarity
from collections import defaultdict
from gregorian import Granularities as G, TimePoint, TimeInterval, Scope
from RotatE._model import KGEModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TKGEModel(nn.Module):
    def __init__(self, model, entity_embedding, relation_embedding, scope, embedding_range=None,
                 gamma=None, modulus=None, temporal_entities=-1, temporal_relations=-1):
        """
        We instantiate linear parameter matric for each time slice of each time granularity levels.
        """
        super(TKGEModel, self).__init__()
        self.size = {}
        self.scope = scope
        self.gamma = nn.Parameter(gamma, requires_grad=False)
        self.modulus = nn.Parameter(modulus, requires_grad=False)
        self.embedding_range = nn.Parameter(embedding_range, requires_grad=False)
        self.entity_embedding = nn.Parameter(entity_embedding, requires_grad=False)
        self.relation_embedding = nn.Parameter(relation_embedding, requires_grad=False)
        self.temporal_entities = temporal_entities if temporal_entities > 0 else self.entity_embedding.size(0)
        self.temporal_relations = temporal_relations if temporal_relations > 0 else self.relation_embedding.size(0)
        self.scoring = {
            'TransE':   KGEModel.TransE,
            'DistMult': KGEModel.DistMult,
            'ComplEx':  KGEModel.ComplEx,
            'RotatE':   KGEModel.RotatE,
            'pRotatE':  KGEModel.pRotatE
        }[model]
        self.edim = self.entity_embedding.size()[1]
        self.rdim = self.relation_embedding.size()[1]
        self.e_layer = nn.ParameterDict()
        self.r_layer = nn.ParameterDict()
        for l in G.each(self.scope.levels):
            self.size[l] = self.scope.size(granularities=l)
            e_layer = nn.Parameter(torch.ones(self.size[l], self.edim).diag_embed(dim1=1).view(-1, self.edim))
            r_layer = nn.Parameter(torch.ones(self.size[l], self.rdim).diag_embed(dim1=1).view(-1, self.rdim))
#            e_layer = nn.Parameter(torch.zeros(self.size[l]*self.edim, self.edim))
#            nn.init.uniform_(tensor=e_layer, a=-1, b=1)
#            r_layer = nn.Parameter(torch.zeros(self.size[l]*self.rdim, self.rdim))
#            nn.init.uniform_(tensor=r_layer, a=-1, b=1)

            self.e_layer[l.name] = e_layer
            self.r_layer[l.name] = r_layer
            logging.debug('Parameter[%s] size: %s' % (l.name, self.size[l]))

    def scoring_function(self, s, p, o, mode='single', activate=False):
        result = self.scoring(s, p, o, mode=mode, embedding_range=self.embedding_range, gamma=self.gamma, modulus=self.modulus)
        if activate:
            result = self.activate(result)
        return result

    def activate(self, x, gamma=1):
        return 1/(torch.abs(x/self.gamma.item())+1)

    def predict_time(self, sample, phase, mode=None, entities_only=False):
        batch_size = sample.size(0)

        s = self.entity_embedding.index_select(dim=0,   index=sample[:,0])
        p = self.relation_embedding.index_select(dim=0, index=sample[:,1])
        o = self.entity_embedding.index_select(dim=0,   index=sample[:,2])
        offset = 0

        # s = self._forward(s, levels, time_filter)
        # o = self._forward(o, levels, time_filter)
        # if not entities_only:
        #     p = self._forward(p, levels, time_filter, on_rel=True)

        result = torch.zeros(batch_size, self.scope.size()).to(device)
        for l in G.each(self.scope.levels):
            sz = self.size[l]
            s = linear(s, self.e_layer[l.name]).view(batch_size, sz, -1, self.edim)
            o = linear(o, self.e_layer[l.name]).view(batch_size, sz, -1, self.edim)
            if not entities_only:
                p = linear(p, self.r_layer[l.name]).view(batch_size, sz, -1, self.rdim)

            scores = self.compute_scores(s, p, o, l)
            activ = self.activate(scores)
            if mode == 'min':
                t = activ-.5
                t = (torch.arange(t.size(1))+1).repeat(t.size(0), 1).to(device)*(t>0).long()
                t[t==0]=t.size(1)+1
                activ = t.min(dim=1, keepdim=True)[1][:, 0, None]
                activ = activ.unsqueeze(dim=1)
            elif mode == 'max':
                t = activ-.5
                t = (torch.arange(t.size(1))+1).repeat(t.size(0), 1).to(device)*(t>0).long()
                activ = t.max(dim=1, keepdim=True)[1][:, 0, None]
                activ = activ.unsqueeze(dim=1)
            else:
                activ = activ.argmax(dim=1)
            activ = activ.view(-1)
            filte = activ.unsqueeze(dim=1).repeat(1, self.edim).unsqueeze(dim=1)
            filtr = activ.unsqueeze(dim=1).repeat(1, self.rdim).unsqueeze(dim=1)
            s = s.view(batch_size, sz, -1).gather(dim=1, index=filte).view(batch_size, -1, self.edim)
            o = o.view(batch_size, sz, -1).gather(dim=1, index=filte).view(batch_size, -1, self.edim)
            if not entities_only:
                p = p.view(batch_size, sz, -1).gather(dim=1, index=filtr).view(batch_size, -1, self.rdim)
            
            result[torch.arange(activ.size(0)), offset+activ]=1
            offset += sz
        return result

    def _forward(self, x, levels, time_filter, phase=None, on_rel=False):
        batch_size = x.size(0)
        result = x
        offset = 0
        dim = (self.rdim if on_rel else self.edim)
        for l in G.each(levels):
            sz = self.size[l]
            layer = (self.r_layer if on_rel else self.e_layer)[l.name]
            result = linear(result, layer)
            if phase == l:
                return result.view(batch_size, sz, -1, dim)

            result = result.view(batch_size, sz, -1)
            result = time_filter(offset, sz, result)
            result = result.view(batch_size, -1, dim)
            offset += sz
        return result

    def forward(self, xy, phase, entities_only=False):
        assert phase&self.scope.levels, "Unspecified level"
        spo, time, nh, nt = xy
        def time_filter(offset, sz, result):
            adj_time = time[:, offset:offset+sz]
            filt = adj_time.nonzero()[:,1]
            return result.gather(dim=1, index=filt.unsqueeze(dim=1).repeat(1, result.size(2)).unsqueeze(dim=1))
        s = self.entity_embedding.index_select(dim=0,   index=spo[:,0])
        p = self.relation_embedding.index_select(dim=0, index=spo[:,1])
        o = self.entity_embedding.index_select(dim=0,   index=spo[:,2])

        s = self._forward(s, self.scope.levels, time_filter, phase=phase)
        o = self._forward(o, self.scope.levels, time_filter, phase=phase)
        if entities_only:
            p = p.unsqueeze(dim=1).repeat(1, self.size[phase], 1, 1)
        else:
            p = self._forward(p, self.scope.levels, time_filter, phase=phase, on_rel=True)
        h, t = None, None
        if not nh is None and nh.size(1) > 0: 
            nhs = nh.size(1)
            h = self.entity_embedding.index_select(dim=0, index=nh.view(-1)).view(spo.size(0), nhs, -1)
            h = self._forward(h, self.scope.levels, time_filter, phase=phase).view(s.size(0), s.size(1), -1, s.size(3))
        if not nt is None and nt.size(1) > 0:
            nts = nt.size(1)
            t = self.entity_embedding.index_select(dim=0, index=nt.view(-1)).view(spo.size(0), nts, -1)
            t = self._forward(t, self.scope.levels, time_filter, phase=phase).view(s.size(0), s.size(1), -1, s.size(3))
        return s, p, o, h, t

    def compute_scores(self, s, p, o, phase, mode='single', activate=False):
        result = []
        for i in range(self.size[phase]):
            score = self.scoring_function(s[:, i, :, :], p[:, i, :, :], o[:, i, :, :], mode=mode, activate=activate)
            result.append(score)
        return torch.stack(result, dim=1).squeeze()

    def freeze_layers(self, phase, incremental=False):
        # Backpropagate only on the phase-relevant layers.
        for l in G.each(self.scope.levels):
            if phase == l or (phase < l and incremental):
                self.e_layer[l.name].requires_grad = True
                self.r_layer[l.name].requires_grad = True
            else:
                self.e_layer[l.name].requires_grad = False
                self.r_layer[l.name].requires_grad = False
                self.e_layer[l.name].grad = None
                self.r_layer[l.name].grad = None

     
