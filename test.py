#!/usr/bin/python3
import logging, sys
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from experiment import parse_args, Experiment
from gregorian import Granularities as G, TimePoint, TimeInterval, Scope
from dataloader import TemporalTrainingDataset, TemporalTestDataset, UniformSampleIterator, RampUpSampleIterator
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, average_precision_score, balanced_accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def run(exp, examples=None, test_levels=False, test_time=False,
        test_ranking=False, test_scoping=False, scores_filename='TKGModel_test_scores'):
    args = exp.args
    scope = args.scope
    model = exp.model
    if not examples:
        examples = exp.test_examples

    test_levels  |= args.test_levels
    test_time    |= args.test_time
    test_ranking |= args.test_ranking
    test_scoping |= args.test_scoping
    if not (test_levels | test_time | test_ranking | test_scoping):
        logging.info("Force default test mode")
        test_levels=True

    model.eval()

    test_datasets = {}
    step, total_steps = 0, 0
    for phase in G.each(scope.levels):
        print(phase)
        ds = DataLoader(
            TemporalTestDataset(
                examples, 
                exp.triples2time,
                model.temporal_entities, 
                model.temporal_relations, 
                scope,
                phase
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TemporalTestDataset.collate_fn
        )
        test_datasets[phase] = ds
        total_steps += len(ds)

    metrics = {}
    with torch.no_grad():
        for phase in G.each(scope.levels):
            test_dataset = test_datasets[phase]
            step += len(test_dataset)
            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

            sample_dates = None
            if test_ranking or test_time:
                if args.time_sample > 0:
                    sample_dates = model.scope.midpoints(phase, args.time_sample)
                else:
                    sample_dates = model.scope.sample(phase, -args.time_sample)
                logging.debug([str(x) for x in sample_dates])
            logs = test_step(exp, test_dataset, triples2time=exp.triples2time,
                sample_dates=sample_dates,
                test_levels=test_levels, test_time=test_time,
                test_ranking=test_ranking, test_scoping=test_scoping,
                scores_filename=scores_filename
            )
            # print('[phase]', phase)
            # print('[logs[phase]]')
            # print(logs[phase])

            metrics[phase] = {
                "dummy" : 0
            }
            if test_levels:
                metrics[phase] = {
                    **metrics[phase],
                    "AUC-ROC" : roc_auc_score(logs[phase][1], logs[phase][2]),
                    "Bal-Acc" : balanced_accuracy_score(logs[phase][1], logs[phase][3]),
                    "Pre."   : precision_score(logs[phase][1], logs[phase][3]),
                    "Rec."  : recall_score(logs[phase][1], logs[phase][3]),
                    "F1"      : f1_score(logs[phase][1], logs[phase][3]),
                }
                metrics[phase].pop("dummy", None)
            if test_time:
                metrics[phase] = {
                    **metrics[phase],
                    "T-AUC-ROC" : roc_auc_score(logs[phase][4], logs[phase][5]),
                    "T-Bal-Acc" : balanced_accuracy_score(logs[phase][4], logs[phase][6]),
                    "T-F1"      : f1_score(logs[phase][4], logs[phase][6]),
                    "T-Pre."     : precision_score(logs[phase][4], logs[phase][6]),
                    "T-Rec."     : recall_score(logs[phase][4], logs[phase][6]),
                }
                metrics[phase].pop("dummy", None)
            if test_ranking:
                metrics[phase] = {
                    **metrics[phase],
                    "Pos. Rerank Error" : logs[phase][9]/float(max(1, 2*logs[phase][7])),
                    "Neg. Rerank Error" : logs[phase][10]/float(max(1, 2*logs[phase][8])),
                    "Mean Rank Delta"   : logs[phase][11]/float(max(1, 2*logs[phase][7])),
                    "HITS@1"  : logs[phase][24]/float(max(1, logs[phase][7]*2)),
                    "HITS@3"  : logs[phase][25]/float(max(1, logs[phase][7]*2)),
                    "HITS@5"  : logs[phase][26]/float(max(1, logs[phase][7]*2)),
                    "HITS@10" : logs[phase][27]/float(max(1, logs[phase][7]*2)),
                    "MR"      : logs[phase][28]/float(max(1, logs[phase][7]*2)),
                    "MRR"     : logs[phase][29]/float(max(1, logs[phase][7]*2)),
                } 
                metrics[phase].pop("dummy", None)
            if test_scoping:
                metrics[phase] = {
                    **metrics[phase],
                    "Mean Jaccard"    : logs[phase][15]/float(max(1, logs[phase][21])),
                    "Scoping Accuracy": logs[phase][22]/float(max(1, logs[phase][21])),
                    "Scoping Distance": logs[phase][23]/float(max(1, logs[phase][21])),
                    "Random Accuracy" : logs[phase][30]/float(max(1, logs[phase][21])),
                }
                metrics[phase].pop("dummy", None)
    return metrics

def test_step(exp, test_dataset, sample_dates=None, triples2time=None,
              test_levels=False, test_time=False, test_ranking=False,
              test_scoping=False, scores_filename='TKGModel_test_scores'):
    '''
    Evaluate the model on test or valid datasets
    '''
    batch = 0
    logs = defaultdict(lambda: [ 0, [], [], [], [], [], [], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0., 0. ])
    model = exp.model
    args = exp.args
    scores_list = []
    # print('[test_dataset.dataset]', type(test_dataset.dataset), len(test_dataset.dataset))

    for pos_sample, pos_interval, phase in test_dataset:
        # print('[phase]', phase)
        # print('[pos_sample, pos_interval]', pos_sample.shape, pos_interval.shape)
        # print('[pos_sample]', pos_sample)
        # print('[pos_interval]', pos_interval)
        pos_sample = pos_sample.to(device)
        pos_interval = pos_interval.to(device)

        batch_size = pos_sample.size(0)
        prefix = phase.prefix(model.scope.levels)
        offset1 = 0 if prefix is None else model.scope.size(prefix)
        offset2 = offset1 + model.scope.size(phase)

        logs[phase][0] += 1
        logs[phase][12] += pos_sample.size(0)
        if test_levels:
            s, p, o, _, _ = model((pos_sample, pos_interval, None, None), phase, args.entities_only)
            # save scores
            # print('[s, p, o]', s.shape, p.shape, o.shape)
            scores = model.compute_scores(s, p, o, phase)
            # print('[scores.shape]', scores.shape)
            scores_list.append(scores.detach().cpu().numpy())
            predicted_times = model.activate(scores)

            if args.display:
                display(model, pos_sample, pos_interval, predicted_times, phase, offset1, offset2)
            
            y_true = (pos_interval[:,offset1:offset2]).int().cpu().numpy().flatten()
            y_pred = predicted_times.cpu().numpy().flatten()

            logs[phase][0] += 1
            logs[phase][1].extend(y_true)
            logs[phase][2].extend(y_pred)
            logs[phase][3].extend((y_pred > .5).astype(int))

        if test_ranking or test_time or test_scoping:
            pos_sample = pos_sample.unique(dim=0)
            batch_size = pos_sample.size(0) 

        if test_scoping:
            predicted_starts = model.predict_time(pos_sample, phase, mode="min")
            predicted_ends = model.predict_time(pos_sample, phase, mode="max")
            predicted_best = model.predict_time(pos_sample, phase)
            random_guess =  torch.stack([model.scope.pick(phase, max_tries=50, raw=True) for spo in enumerate(pos_sample)])
            scoping_sample = 0
            for i, spo in enumerate(pos_sample):
                s, p, o = spo[0].item(), spo[1].item(), spo[2].item()
                st = model.scope.interpret(predicted_starts[i])
                en = model.scope.interpret(predicted_ends[i])
                b  = model.scope.interpret(predicted_best[i])
                rnd = model.scope.interpret(random_guess[i])
                truths = triples2time[(s,p,o)]
                if i > 0 and (s,p,o) == (pos_sample[i-1,0].item(), pos_sample[i-1,1].item(), pos_sample[i-1,2].item()):
                    continue
                logs[phase][21] += 1
                interval = model.scope.clip(st, en)
                truth = model.scope.clip(min({t.start for t in truths}), max({t.end for t in truths}))
                
                jacc = "N/A"
                # if st or en:
                #     jacc =  model.scope.jaccard(interval, truth, phase)
                #     logs[phase][15] += jacc
                hit = any({b.within(t, down_to=phase) for t in truths})
                if hit:
                    logs[phase][22] += 1
                random_hit = any({rnd.within(t, down_to=phase) for t in truths})
                if random_hit:
                    logs[phase][30] += 1
                dist = min({TimeInterval(t.mean().truncate(model.scope.start, reverse=True), b).delta(phase) for t in truths})
                logs[phase][23] += dist
                if args.display:
                    print(s, p, o, phase, {str(x) for x in truths}, truth, interval, jacc, b, hit, dist)
            
        if test_ranking:
            s = model.entity_embedding.index_select(dim=0,   index=pos_sample[:,0]).unsqueeze(dim=1)
            p = model.relation_embedding.index_select(dim=0, index=pos_sample[:,1]).unsqueeze(dim=1)
            o = model.entity_embedding.index_select(dim=0,   index=pos_sample[:,2]).unsqueeze(dim=1)
            e = model.entity_embedding.unsqueeze(dim=0).repeat(batch_size, 1, 1)

            nts_scores = model.scoring_function(e, p, o, mode='tail-batch', activate=True)
            nto_scores = model.scoring_function(s, p, e, mode='head-batch', activate=True)
            nts_argsort = torch.argsort(nts_scores, dim=1, descending=True)
            nto_argsort = torch.argsort(nto_scores, dim=1, descending=True)
            nts_ranking = (nts_argsort == pos_sample[:, 0].unsqueeze(dim=1)).nonzero()[:,1]+1
            nto_ranking = (nto_argsort == pos_sample[:, 2].unsqueeze(dim=1)).nonzero()[:,1]+1
            assert nts_ranking.size(0) == batch_size and nto_ranking.size(0) == batch_size

        if test_ranking or test_time:
            levels = phase if prefix is None else prefix|phase
            for d in sample_dates:
                valid_triples = []
                dv = d.vectorize(model.scope)

                def time_filter(offset, sz, result):
                    adj_time = dv[offset:offset+sz]
                    filt = torch.from_numpy(adj_time.nonzero()[0]).to(device)
                    return result.gather(dim=1, index=filt.unsqueeze(dim=0).unsqueeze(dim=0).repeat(batch_size, 1, result.size(2)))

                for i in range(batch_size): 
                    spo = (pos_sample[i, 0].item(), pos_sample[i, 1].item(), pos_sample[i, 2].item())
                    times = test_dataset.dataset.valid_times[spo]
                    valid_triples.append(1 if times and any([t.contains(d) for t in times]) else 0)

                valid_triples = torch.LongTensor(valid_triples).to(device).view(-1)

                s = model.entity_embedding.index_select(dim=0,   index=pos_sample[:,0]).unsqueeze(dim=1)
                p = model.relation_embedding.index_select(dim=0, index=pos_sample[:,1]).unsqueeze(dim=1)
                o = model.entity_embedding.index_select(dim=0,   index=pos_sample[:,2]).unsqueeze(dim=1)
                s = model._forward(s, levels, time_filter)
                o = model._forward(o, levels, time_filter)
                if not args.entities_only:
                    p = model._forward(p, levels, time_filter, on_rel=True)

                y_scores = model.scoring_function(s, p, o).view(-1)
                y_pred = model.activate(y_scores).view(-1)
                if test_time and len(s.size()) == 3:
                    y_pred = y_pred.cpu().numpy()
                    y_true = valid_triples.cpu().numpy()

                    if args.display:
                        display2(model, d, pos_sample, y_true, y_pred)
                    
                    logs[phase][4].extend(y_true)
                    logs[phase][5].extend(y_pred)
                    logs[phase][6].extend((y_pred > .5).astype(int))

                if test_ranking:
                    def time_filter(offset, sz, result):
                        adj_time = dv[offset:offset+sz]
                        filt = torch.from_numpy(adj_time.nonzero()[0]).to(device) 
                        result = result.gather(dim=1, index=filt.unsqueeze(dim=0).unsqueeze(dim=0).repeat(model.entity_embedding.size(0), 1, result.size(2)))
                        return result

                    e = model._forward(model.entity_embedding, levels, time_filter).squeeze(dim=1)
                    e = e.unsqueeze(dim=0).repeat(batch_size, 1, 1)
                    ts_scores = model.scoring_function(e, p, o, mode='tail-batch', activate=True)
                    to_scores = model.scoring_function(s, p, e, mode='head-batch', activate=True)
                    ts_argsort = torch.argsort(ts_scores, dim=1, descending=True)
                    to_argsort = torch.argsort(to_scores, dim=1, descending=True)
                    ts_ranking = (ts_argsort == pos_sample[:, 0].unsqueeze(dim=1)).nonzero()[:,1]+1
                    to_ranking = (to_argsort == pos_sample[:, 2].unsqueeze(dim=1)).nonzero()[:,1]+1
                    #print(nts_ranking, ts_ranking, valid_triples, (ts_ranking <= nts_ranking).long(), valid_triples*(ts_ranking <= nts_ranking).long())
                    assert ts_ranking.size(0) == batch_size and to_ranking.size(0) == batch_size

                    invalid_triples  = (valid_triples == 0).long()
                    logs[phase][7]  += valid_triples.sum()
                    logs[phase][8]  += invalid_triples.sum()
                    logs[phase][9]  += (valid_triples*(ts_ranking > nts_ranking).long()).sum().item()
                    logs[phase][9]  += (valid_triples*(to_ranking > nto_ranking).long()).sum().item()
                    logs[phase][10] += (invalid_triples*(ts_ranking <= nts_ranking).long()).sum().item()
                    logs[phase][10] += (invalid_triples*(to_ranking <= nto_ranking).long()).sum().item()
                    logs[phase][11] += (valid_triples*nts_ranking-valid_triples*ts_ranking).sum().item()
                    logs[phase][11] += (valid_triples*nto_ranking-valid_triples*to_ranking).sum().item()

                    logs[phase][24]  += (valid_triples*(ts_ranking <=  1).long()).sum().item()
                    logs[phase][24]  += (valid_triples*(to_ranking <=  1).long()).sum().item()
                    logs[phase][25]  += (valid_triples*(ts_ranking <=  3).long()).sum().item()
                    logs[phase][25]  += (valid_triples*(to_ranking <=  3).long()).sum().item()
                    logs[phase][26]  += (valid_triples*(ts_ranking <=  5).long()).sum().item()
                    logs[phase][26]  += (valid_triples*(to_ranking <=  5).long()).sum().item()
                    logs[phase][27]  += (valid_triples*(ts_ranking <= 10).long()).sum().item()
                    logs[phase][27]  += (valid_triples*(to_ranking <= 10).long()).sum().item()
                    logs[phase][28]  += (valid_triples*(ts_ranking).long()).sum().item()
                    logs[phase][28]  += (valid_triples*(to_ranking).long()).sum().item()
                    logs[phase][29]  += (valid_triples.float()*(1./ts_ranking.float())).sum().item()
                    logs[phase][29]  += (valid_triples.float()*(1./to_ranking.float())).sum().item()


                    y_pred = (model.activate(y_scores).view(-1) > .5).long()

                    valid_triples *= y_pred
                    logs[phase][13] += (valid_triples*nts_ranking-valid_triples*ts_ranking).sum().item()
                    logs[phase][13] += (valid_triples*nto_ranking-valid_triples*to_ranking).sum().item()
                    valid_triples = y_pred
                    logs[phase][14] += (valid_triples*nts_ranking-valid_triples*ts_ranking).sum().item()
                    logs[phase][14] += (valid_triples*nto_ranking-valid_triples*to_ranking).sum().item()


        batch += 1
        #print(batch, logs[phase])
    # print('[scores_list]', scores_list)
    scores_list = np.concatenate(scores_list)
    print('[scores_list]', scores_list.shape)
    np.save(scores_filename, scores_list)
    return logs

def display(model, sample, ground_truth, predicted_times, phase, offset1, offset2, edict=None, rdict=None):
    copy = ground_truth.clone()
    copy[:,offset1:offset2]=(predicted_times>.5)
    for i, spo in enumerate(sample):
        s, p, o = spo[0].item(), spo[1].item(), spo[2].item()
        if edict and rdict:
            s, p, o = edict.inverse[s], rdict.inverse[p], edict.inverse[o]
        prediction = copy[i, offset1:offset2]
        grnd_truth = ground_truth[i, offset1:offset2]
        print(phase, s, p, o,
              predicted_times[i],
              prediction,
              grnd_truth,
              model.scope.interpret(copy[i]),
              model.scope.interpret(ground_truth[i]))

def display2(model, date, sample, y_true, y_pred):
    for i, spo in enumerate(sample):
        print(date, spo[0].item(), spo[1].item(), spo[2].item(), y_true[i], y_pred[i])
    
if __name__ == '__main__':
    args = parse_args()
    exp = Experiment(args)
    if args.do_valid:
        print('[1]')
        metrics = run(exp, examples=exp.valid_examples, scores_filename='TKGModel_valid_scores')
    elif args.evaluate_train:
        print('[2]')
        metrics = run(exp, examples=exp.train_examples, scores_filename='TKGModel_train_scores')
    else:
        print('[3]')
        metrics = run(exp, scores_filename='TKGModel_test_scores')
    exp.log_metrics('Test', metrics)
