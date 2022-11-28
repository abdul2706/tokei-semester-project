import os, logging
import test
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax, logsigmoid, relu
from experiment import parse_args, Experiment
from gregorian import Granularities as G
from dataloader import TemporalTrainingDataset, TemporalTestDataset, UniformSampleIterator, RampUpSampleIterator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def similarity_loss(model, phase, truth, sample, scores):
    ss = model.entity_embedding.index_select(dim=0,   index=sample[:,0]).unsqueeze(dim=1)
    pp = model.relation_embedding.index_select(dim=0, index=sample[:,1]).unsqueeze(dim=1)
    oo = model.entity_embedding.index_select(dim=0,   index=sample[:,2]).unsqueeze(dim=1)
    return truth * relu(scores-model.scoring_function(ss, pp, oo, phase).repeat(1, scores.size(1)))

def compute_loss(model, scores, truth):
    return (truth-model.activate(scores))**2

def non_temporal_score(spo, model):
    s = model.entity_embedding.index_select(dim=0,   index=spo[:,0]).unsqueeze(dim=1)
    p = model.relation_embedding.index_select(dim=0, index=spo[:,1]).unsqueeze(dim=1)
    o = model.entity_embedding.index_select(dim=0,   index=spo[:,2]).unsqueeze(dim=1)
    return model.scoring_function(s, p, o)

def train_step(model, optimizer, train_iterator, args):
    '''
    A single train step. Apply back-propation and return the loss
    '''
    model.train()

    optimizer.zero_grad()
    pos_sample, pos_intervals, neg_heads, neg_tails, phase = next(train_iterator)

    offset = model.scope.prefix_size(phase)
    size   = model.scope.size(granularities=phase)

    logging.debug('TrainStep(phase=%s)' % phase.name)
    model.freeze_layers(phase, args.incremental)

    pos_sample = pos_sample.to(device)
    pos_intervals = pos_intervals.to(device)
    neg_heads = neg_heads.to(device)
    neg_tails = neg_tails.to(device)

    ground_truth = pos_intervals[:,offset:offset+size]
    ps, pp, po, nh, nt = model((pos_sample, pos_intervals, neg_heads, neg_tails), phase, args.entities_only)
    pos_scores = model.compute_scores(ps, pp, po, phase)
    pos_loss = compute_loss(model, pos_scores, ground_truth)
    if args.cost == 0:
        pos_loss = logsigmoid(pos_loss)
    pos_loss = pos_loss.mean()
    neg_loss = torch.tensor(0.0).to(device)
    if args.negative_sample_size > 0:
        nh_scores = model.compute_scores(nh, pp, po, phase, mode='head-batch')
        nt_scores = model.compute_scores(ps, pp, nt, phase, mode='tail-batch')
        neg_scores = torch.cat((nh_scores, nt_scores), dim=2)
        neg_loss = compute_loss(model, neg_scores, 0)
        if args.cost == 0:
            neg_loss = logsigmoid(neg_loss)
        if args.adversarial_temperature > 0.0:
            neg_loss = (1-(softmax(neg_scores * args.adversarial_temperature, dim = 2).detach()) * neg_loss).sum(dim=2)
        else:
            neg_loss = neg_loss.mean(dim=2)
        if args.cost == 0:
            neg_loss = -neg_loss
        neg_loss = neg_loss.mean()

    sim_loss = torch.tensor(0.0).to(device)
    if args.similarity_loss_coef > 0.0:
        if args.cost != 0:
            sim_loss = args.similarity_loss_coef*similarity_loss(model, phase,
                        ground_truth, pos_sample, pos_scores)
        sim_loss = sim_loss.mean()                 

    loss = (pos_loss+neg_loss+sim_loss)
    loss.backward()
    optimizer.step()

    log = {
            'pos_sample_loss': pos_loss.item(),
            'neg_sample_loss': neg_loss.item(),
            'sim_sample_loss': sim_loss.item(),
            'loss': loss.item()
    }

    return log

def run(exp):
    args = exp.args
    model = exp.model
    if args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    nentity = len(exp.entity_ids)
    nrelation = len(exp.relation_ids)

    train_dataloaders = []
    for phase in G.each(exp.args.levels):
        # Set training dataloader iterator
        train_dataloaders.append(DataLoader(
            TemporalTrainingDataset(
                exp.train_examples, exp.triples2time,
                model.temporal_entities, model.temporal_relations,
                model.scope, phase, negative_sample_size=args.negative_sample_size), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TemporalTrainingDataset.collate_fn
        ))

    if args.ramp_up > 1:
        train_iterator = RampUpSampleIterator(train_dataloaders, args.phase_length, args.ramp_up)
    else:
        train_iterator = UniformSampleIterator(train_dataloaders, args.phase_length)

    # Set training configuration
    current_learning_rate = args.learning_rate
    optimizer = exp.load_optimizer(current_learning_rate)
    if args.warm_up_steps:
        warm_up_steps = args.warm_up_steps
    else:
        warm_up_steps = args.max_steps // 2

    init_step = args.init_step

    # Set valid dataloader as it would be evaluated during training
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('ramp_up = %f' % args.ramp_up)
    logging.info('regularization = %f' % args.regularization)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('phase_length = %d' % args.phase_length)
    logging.info('learning_rate = %f' % args.learning_rate)
    logging.info('similarity cost coef = %f' % args.similarity_loss_coef)
    logging.info('negative_sample_size = %f' % args.negative_sample_size)
    logging.info('negative_adversarial_sampling = %s' % args.negative_adversarial_sampling)
    logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    training_logs = []

    logging.info('Evaluating on Validation Dataset...')
    metrics = test.run(exp, exp.valid_examples)
    exp.log_metrics('Valid', metrics)
    # Training Loop
    for step in range(init_step, args.max_steps+1):
        if train_iterator.step <= 0:
            # Set training configuration
            current_learning_rate = args.learning_rate

        log = train_step(model, optimizer, train_iterator, args)
                    
        training_logs.append(log)

        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
            exp.log_metrics('Training average', metrics, step)
            training_logs = []

        if step % args.valid_steps == 0:
            logging.info('Evaluating on Validation Dataset...')
            metrics = test.run(exp, exp.valid_examples)
            exp.log_metrics('Valid', metrics, step)
                
        if args.warm_up_steps:
            if step%args.phase_length == warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                exp.load_optimizer(current_learning_rate)
            elif step%args.phase_length == 0:
                current_learning_rate = args.learning_rate
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                exp.load_optimizer(current_learning_rate)
                #warm_up_steps = step + args.warm_up_steps * args.ramp_up

        if step % args.save_checkpoint_steps == 0 or step == args.max_steps - 1:
            exp.save_model(optimizer, {
                'step': step, 
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
            })

    logging.info('Evaluating on Test Dataset...')
    metrics = test.run(exp, examples=exp.test_examples, test_levels=True, test_time=True)
    exp.log_metrics('Test', metrics, step)

if __name__ == '__main__':
    args = parse_args()
    print('[args]', args)
    run(Experiment(args))

