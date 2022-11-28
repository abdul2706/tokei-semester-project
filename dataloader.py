#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from itertools import product

class TemporalTrainingDataset(Dataset):

    def __init__(self, triples, valid_times, nentity, nrelation, scope, phase, negative_sample_size=None):
        self.len = len(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(set(triples))
        self.triples = triples
        self.size = scope.size(phase)
        self.scope = scope
        self.phase = phase
        self.nentity = nentity
        self.nrelation = nrelation
        self.valid_times = valid_times
        self.negative_sample_size = negative_sample_size
        # self.count = self.count_frequency(quads)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        interval = self.valid_times[(head, relation, tail)]

        positive_sample = torch.LongTensor([head, relation, tail])
        intervals = torch.from_numpy(self.scope.vectorize(interval, self.phase)).float()


        # subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        # subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight])
        neg_heads_size = int(self.negative_sample_size/2)
        neg_tails_size = self.negative_sample_size-neg_heads_size

        negative_heads, negative_tails = torch.tensor([]), torch.tensor([])
        if neg_heads_size > 0:
            negative_sample_list = []
            #negative_sample_size = 0
            while len(negative_sample_list) < neg_heads_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size)
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
                for s in negative_sample[mask]:
                    i = self.valid_times[(s, relation, tail)]
                    if (not i) or not any(x.overlaps(y) for (x, y) in product(i, interval)):
                        negative_sample_list.append(s)
                #negative_sample_size += negative_sample.size
            negative_sample = np.array(negative_sample_list)[:neg_heads_size]
            negative_heads = torch.from_numpy(negative_sample)

        if neg_tails_size > 0:
            negative_sample_list = []
            #negative_sample_size = 0
            while len(negative_sample_list) < neg_tails_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size)
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
                for t in negative_sample[mask]:
                    i = self.valid_times[(head, relation, t)]
                    if (not i) or not any(x.overlaps(y) for (x, y) in product(i, interval)):
                        negative_sample_list.append(t)

                #negative_sample = negative_sample[mask]
                #negative_sample_list.append(negative_sample)
                #negative_sample_size += negative_sample.size
            negative_sample = np.array(negative_sample_list)[:neg_tails_size]
            negative_tails = torch.from_numpy(negative_sample)
        
        return positive_sample, intervals, negative_heads, negative_tails, self.phase
    
    @staticmethod
    def collate_fn(data):
        pos_samples, pos_intervals, neg_heads, neg_tails = [], [], [], []
        for p_sample, interval, n_heads, n_tails, phase in data:
            for i in range(interval.size(0)):
                pos_samples.append(p_sample)
                pos_intervals.append(interval[i,:])
                neg_heads.append(n_heads)
                neg_tails.append(n_tails)
        
        pos_samples = torch.stack(pos_samples, dim=0)
        pos_intervals = torch.stack(pos_intervals, dim=0)
        neg_heads = torch.stack(neg_heads, dim=0)
        neg_tails = torch.stack(neg_tails, dim=0)
        return pos_samples, pos_intervals, neg_heads, neg_tails, data[0][4]

    @staticmethod
    def get_true_head_and_tail(triples):
        true_head = {}
        true_tail = {}
        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

class TemporalTestDataset(Dataset):
    def __init__(self, triples, valid_times, nentity, nrelation, scope, phase):
        self.len = len(triples)
        self.size = scope.size(phase)
        self.scope = scope
        self.phase = phase
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.valid_times = valid_times
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        intervals = self.valid_times[(head, relation, tail)]
        sample = torch.LongTensor([head, relation, tail])
        positive_interval = torch.from_numpy(self.scope.vectorize(intervals, self.phase)).float()
        return sample, positive_interval, self.phase
    
    @staticmethod
    def collate_fn(data):
        pos_samples, pos_intervals = [], []
        for p_sample, p_interval, _ in data:
            for i in range(p_interval.size(0)):
                pos_samples.append(p_sample)
                pos_intervals.append(p_interval[i,:])

        phase = data[0][2]
        pos_samples = torch.stack(pos_samples, dim=0)
        pos_intervals = torch.stack(pos_intervals, dim=0)
        return pos_samples, pos_intervals, phase

class UniformSampleIterator(object):
    def __init__(self, datasets, subphase):
        self.iterators = []
        for d in datasets:
            self.iterators.append(self.one_shot_iterator(d))
        self.step = -1
        self.subphase = subphase
        
    def __next__(self):
        self.step += 1
        index = int(self.step / self.subphase)
        if index >= len(self.iterators):
            self.step = -1
        return next(self.iterators[index])
    
    @staticmethod
    def one_shot_iterator(datasets):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in datasets:
                yield data

class RampUpSampleIterator(object):
    def __init__(self, datasets, phaselen, rampup):
        self.iterators = []
        for d in datasets:
            self.iterators.append(self.one_shot_iterator(d))
        self.step = 0
        self.phaselen = phaselen
        self.rampup = rampup
        self.current = 0
        
    def __next__(self):
        if self.step >= self.phaselen:
            self.step = 0
            self.current += 1
            self.phaselen *= self.rampup
        result = next(self.iterators[self.current])
        self.step += 1
        return result
    
    @staticmethod
    def one_shot_iterator(datasets):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in datasets:
                yield data
