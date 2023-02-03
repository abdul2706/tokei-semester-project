#!/usr/bin/python3
import os
import json
import glob
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
from random import shuffle
from model import TKGEModel
from bidict import bidict
from collections import defaultdict
from torch.utils.data import DataLoader
from gregorian import Granularities as G, TimePoint, TimeInterval, Scope
from dataloader import TemporalTrainingDataset, TemporalTestDataset, UniformSampleIterator, RampUpSampleIterator

TAG = '[experiments.py]'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(TAG, 'device:', device)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cpu',    action='store_true', help='use CPU')
    parser.add_argument('--scope',  type=parse_scope,  help='The time scope (format YYYY_MM_DD_HH_mm_ss,YYYY_MM_DD_HH_mm_ss,<level>)')
    parser.add_argument('--levels', type=parse_levels, help='The temporal layers to use. Any combinations of C, D, Y, M, d, h, m, s (eg. "YMd" for years, months and days')

    parser.add_argument('--test_levels',    action='store_true')
    parser.add_argument('--test_ranking',   action='store_true')
    parser.add_argument('--test_time',      action='store_true')
    parser.add_argument('--test_scoping',   action='store_true')
    parser.add_argument('--display',        action='store_true')
    parser.add_argument('--entities_only',  action='store_true')
    parser.add_argument('--do_valid',       action='store_true', help="Validate models")
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--no_dict',        action='store_true', help='Reads the input train/test/valid key directly (i.e. w/o dictionary conversion)')
    parser.add_argument('-df', '--date_format', type=str, default=None, help='If set to "yago", the expected date format is that of YAGO ')

    parser.add_argument('-data','--data_path',  type=str, default=None, required=True)
    
    parser.add_argument('-adv', '--negative_adversarial_sampling',     action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', type=float, default=0.0)
    parser.add_argument('-n', '--negative_sample_size',    type=int,   default=0)
    parser.add_argument('-b', '--batch_size',              type=int,   default=-1)
    parser.add_argument('--test_batch_size',               type=int,   default=-1,    help='Valid/test batch size')
    parser.add_argument('-r', '--regularization',          type=float, default=-1)
    parser.add_argument('-p', '--phase_length',            type=int,   default=2000, help='Number of iterations per phase.')
    parser.add_argument('-ru', '--ramp_up',                type=int,   default=1,    help='Factor by which to increase the number of iterations per phase.')
    parser.add_argument('-sim', '--similarity_loss_coef',  type=float, default=0,    help='Coefficient for the similarity loss term')
    parser.add_argument('-ts', '--time_sample',            type=int,   default=10,    help='Size of the test sample')
    
    parser.add_argument('-lr', '--learning_rate',     type=float, default=0.0001)
    parser.add_argument('-cpu', '--cpu_num',          type=int,   default=10)
    parser.add_argument('-save', '--save_path',       type=str,   default=None)
    parser.add_argument('--init_step',                type=int,   default=1)
    parser.add_argument('--max_steps',                type=int,   default=2000)
    parser.add_argument('--warm_up_steps',            type=int,   default=None)
    parser.add_argument('-base', '--base_model_path', type=str,   default=None, required=True,
                help='If specified, loads a base model from the given path')
    parser.add_argument('-X', '--reinit',             action='store_true',   help='If specified, reinitialized all embedding at start')
    parser.add_argument('-i', '--incremental',        action='store_true')
    parser.add_argument('--cost',                     type=int,   default=0)

    parser.add_argument('--save_checkpoint_steps',  type=int, default=10000)
    parser.add_argument('--log_steps',              type=int, default=1000,   help='train log every xx steps')
    parser.add_argument('--valid_steps',            type=int, default=5000)
    parser.add_argument('--test_log_steps',         type=int, default=10000, help='valid/test log every xx steps')
    parser.add_argument('--seed',                   type=int, default=0,     help='Randomizer seed')

    parser.add_argument('--nentity',         type=int,   help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation',       type=int,   help='DO NOT MANUALLY SET')
    parser.add_argument('--hidden_dim',      type=int,   help='DO NOT MANUALLY SET')
    parser.add_argument('--model',           type=str,   help='DO NOT MANUALLY SET')
    parser.add_argument('--gamma',           type=float, help='DO NOT MANUALLY SET')
    parser.add_argument('--init_model_path', type=str,   help='DO NOT MANUALLY SET')

    return parser.parse_args(args)

def parse_scope(s):
    try:
        s, e, l = s.split(',')
        result = Scope(TimeInterval.parse(s, e), parse_levels(l))
        return result
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def parse_date(s):
    try:
        result = TimePoint.parse(s)
        return result
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def parse_levels(s):
    result = 0
    for c in s:
        if c == 'C':
            result |= G.CENTURY
        elif c == 'D':
            result |= G.DECADE
        elif c == 'Y':
            result |= G.YEAR
        elif c == 'M':
            result |= G.MONTH
        elif c == 'd':
            result |= G.DAY
        elif c == 'h':
            result |= G.HOUR
        elif c == 'm':
            result |= G.MINUTE
        elif c == 's':
            result |= G.SECOND
        else:
            raise ValueError("No such flag: " + c)
    return result

def set_logger(args, log_file=None):
    '''
    Write logs to checkpoint and console
    '''
    if not log_file:
        directory = args.save_path or args.init_model_path or args.base_model_path
        if not os.path.exists(directory):
            os.makedirs(directory)
        if args.save_path:
            log_file = os.path.join(directory, 'train.log')
        else:
            log_file = os.path.join(directory, 'test.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
        )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info("Set logger to %s" % log_file)

# def load_dict(filepath):
#     with open(filepath, encoding='utf-8') as fin:
#         item2id = bidict()
#         i = 0
#         for line in fin:
#             try:
#                 key, value = line.strip().split('\t', 1)
#             except:
#                 raise ValueError(line)
#             try:
#                 item2id[value] = int(key)
#                 i = item2id[value] + 1
#             except ValueError:
#                 item2id[key] = i
#                 i += 1
#     return item2id

def load_dict(filepath):
    with open(filepath, encoding='utf-8') as fin:
        item2id = bidict()
        i = 0
        for line in fin:
            try:
                key, value = line.strip().split('\t', 1)
            except:
                raise ValueError(line)
            item2id[key] = i
            i += 1
    return item2id

def read_triples(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path, encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def read_temporal_triples(filepath, entity2id=None, relation2id=None):
    '''
    Read temporal quads and map them into ids.
    '''
    triples, times = [], defaultdict(set)
    with open(filepath, encoding='utf-8') as fin:
        for line in fin:
            try:
                h, r, t, s, e = line.strip().split('\t')
                key = (int(h) if entity2id is None else entity2id[h],
                       int(r) if relation2id is None else relation2id[r],
                       int(t) if entity2id is None else entity2id[t])
                triples.append(key)
                times[key].add(TimeInterval.parse(s, e))
            except ValueError:
                logging.error(line)
    return triples, times


class Experiment:

    def __init__(self, args):
        TAG2 = TAG + '[Experiment][__init__]'
        self.args = args
        if args.data_path is None:
            raise ValueError('One of data_path must be choosed.')

        set_logger(args)

        if args.cpu:
            logging.info("Forcing CPU usage...")
            device = torch.device("cpu") 

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        logging.debug("Loading dictionaries...")
        self.entity_ids = load_dict(os.path.join(args.data_path, 'entities.dict'))
        self.relation_ids = load_dict(os.path.join(args.data_path, 'relations.dict'))
        print(TAG2, '[self.entity_ids]', len(self.entity_ids))
        print(TAG2, '[self.relation_ids]', len(self.relation_ids))

        logging.debug("Loading data...")
        self.all_examples, self.triples2time = read_temporal_triples(
            os.path.join(args.data_path, 'temporal'),
            entity2id=None   if args.no_dict else self.entity_ids,
            relation2id=None if args.no_dict else self.relation_ids)
        # print(TAG2, '[self.all_examples]', self.all_examples)
        # print(TAG2, '[self.triples2time]', self.triples2time)
        shuffle(self.all_examples)
        split1 = int(len(self.all_examples)*.7)
        split2 = int(len(self.all_examples)*.85)
        self.train_examples = self.all_examples[:split1]
        self.test_examples  = self.all_examples[split1:split2]
        self.valid_examples = self.all_examples[split2:]

        # temporal_entities = len({s for (s,p,o) in self.all_examples } | {o for (s,p,o) in self.all_examples })
        # temporal_relations = len({p for (s,p,o) in self.all_examples})
        # print(TAG2, '[temporal_entities]', temporal_entities)
        # print(TAG2, '[temporal_relations]', temporal_relations)

        logging.debug("Loading model...")
        self.model = self.load_model(
             temporal_entities=len({s for (s,p,o) in self.all_examples } | {o for (s,p,o) in self.all_examples }),
             temporal_relations=len({p for (s,p,o) in self.all_examples})
        )
        logging.info('Base Model: %s (%s)' % (args.model, args.base_model_path))
        logging.info('Data Path:  %s' % args.data_path)
        logging.info('#entity:    %d' % len(self.entity_ids))
        logging.info('#relation:  %d' % len(self.relation_ids))
        logging.info('#train:     %d' % len(self.train_examples))
        logging.info('#valid:     %d' % len(self.valid_examples))
        logging.info('#test:      %d' % len(self.test_examples))
        logging.info('Scope:      %s' % self.model.scope.pretty())
        logging.info('Model Parameter Configuration:')
        for name, param in self.model.named_parameters():
            logging.info('Parameter %s: %s, require_grad = %s' %
                         (name, str(param.size()), str(param.requires_grad)))

    def save_model(self, optimizer, save_variable_list):
        '''
        Save the parameters of the model and the optimizer,
        as well as some other variables such as step and learning_rate
        '''
        logging.info("Saving model to %s..." % str(self.args.save_path))
        arg_dict = vars(self.args)
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:
            json.dump(arg_dict, fjson, default=str)

        torch.save({
            **save_variable_list,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(self.args.save_path, 'checkpoint')
        )

        entity_embedding = self.model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(self.args.save_path, 'entity_embedding'), 
            entity_embedding
        )

        relation_embedding = self.model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(self.args.save_path, 'relation_embedding'), 
            relation_embedding
        )

        for phase in G.each(self.model.scope.levels):
            layer = self.model.e_layer[phase.name].detach().cpu().numpy()
            np.save(os.path.join(self.args.save_path, phase.name + '.e'), layer)
            layer = self.model.r_layer[phase.name].detach().cpu().numpy()
            np.save(os.path.join(self.args.save_path, phase.name + '.r'), layer)


    def load_model(self, temporal_entities=-1, temporal_relations=-1):
        '''
        Load the parameters of the base model and the optimizer,
        as well as some other variables such as step and learning_rate
        '''
        with open(os.path.join(self.args.base_model_path, 'config.json'), 'r') as fjson:
            base_dict = json.load(fjson)
        initial = not 'scope' in base_dict

        self.args.init_model_path = self.args.base_model_path
        if not initial:
            self.args.base_model_path = base_dict['base_model_path']

        entity_embedding = torch.from_numpy(np.load(os.path.join(self.args.base_model_path, 'entity_embedding.npy'))).to(device)
        relation_embedding = torch.from_numpy(np.load(os.path.join(self.args.base_model_path, 'relation_embedding.npy'))).to(device)

        epsilon = 2.0
        gamma = torch.Tensor([base_dict['gamma']])
        embedding_range = torch.Tensor([(gamma.item() + epsilon) / base_dict['hidden_dim']])
        modulus = torch.Tensor([[0.5 * embedding_range.item()]])

        if initial and self.args.reinit:
            # Follow the same initialization as RotatE
            nn.init.uniform_(tensor=entity_embedding,   a=-embedding_range.item(), b=embedding_range.item())
            nn.init.uniform_(tensor=relation_embedding, a=-embedding_range.item(), b=embedding_range.item())

        if not (initial or self.args.scope):
            self.args.scope = parse_scope(base_dict['scope'])

        if not self.args.levels:
            self.args.levels = self.args.scope.levels

        self.args.model = base_dict['model']
        if not self.args.data_path:
            self.args.data_path = base_dict['data_path']
        if not self.args.hidden_dim:
            self.args.hidden_dim = base_dict['hidden_dim']
        if not self.args.gamma:
            self.args.gamma = base_dict['gamma']
        if self.args.negative_sample_size < 0:
            self.args.negative_sample_size = base_dict['negative_sample_size']
        if self.args.negative_adversarial_sampling < 0:
            self.args.negative_adversarial_sampling = base_dict['negative_adversarial_sampling']
        if self.args.adversarial_temperature < 0:
            self.args.adversarial_temperature = base_dict['adversarial_temperature']
        if self.args.batch_size < 0:
            self.args.batch_size = base_dict['batch_size']
        if self.args.test_batch_size < 0:
            self.args.test_batch_size = base_dict['batch_size']
        if self.args.regularization < 0:
            self.args.regularization = base_dict['regularization']
        # if not self.args.uni_weight:
        #     self.args.uni_weight = base_dict['uni_weight']

        model = TKGEModel(self.args.model, entity_embedding, relation_embedding, self.args.scope,
                          embedding_range=embedding_range, gamma=gamma, modulus=modulus,
                          temporal_entities=temporal_entities, temporal_relations=temporal_relations)

        if not initial:
            checkpoint = torch.load(os.path.join(self.args.init_model_path, 'checkpoint'))
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state_dict'])

        if torch.cuda.device_count() > 1:
            print("Using %d GPUs." % torch.cuda.device_count())
            class DataParallel(nn.DataParallel):
                def __getattr__(self, name):
                    try:
                        return super().__getattr__(name)
                    except AttributeError:
                        return getattr(self.module, name)
            model = DataParallel(model)
        model.to(device)
        return model

    def load_optimizer(self, current_learning_rate):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=current_learning_rate,
            weight_decay=self.args.regularization
        )

        if self.args.init_model_path:
            checkpoint = torch.load(os.path.join(self.args.init_model_path, 'checkpoint'), map_location=device)
            try:
                # current_learning_rate = checkpoint['current_learning_rate']
                # warm_up_steps = checkpoint['warm_up_steps']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                logging.warn("Unable to load optimizer")

        return optimizer

    def log_metrics(self, mode, metrics, step=None):
        '''
        Print the evaluation logs
        '''
        for meta_key in metrics.keys():
            msg = '%s %s' % (mode, meta_key)
            if step:
                msg += ' at step %s ' % step
            if meta_key in G:
                for key in metrics[meta_key]:
                    msg += '\t%s: %f' % (key, metrics[meta_key][key])
                logging.info(msg)
            else:
                logging.info('%s: %f' % (msg, metrics[meta_key]))

# python test.py --cpu_num 2 --seed 0 --data_path data/icews14 --base_model_path models/RotatE_icews14 --save_path models/RotatE_icews14_test --scope d1100_1_1,d2030_12_31,CDY --test_levels --test_time --test_scoping --test_ranking --test_batch_size 128 --max_steps 3000 --valid_steps 1000 -a 1.0 -n 128 -b 1024 -p 1000 -ru 1 -lr 0.001 -ts 10 --do_valid
