import numpy as np
import awkward as ak
import tqdm
import time
import torch
import os
import math

from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger

from torch.profiler import profile, record_function, ProfilerActivity

def _flatten_label(label, mask=None):
    if label is not None:
        if label.ndim > 1:
            label = label.view(-1)
            if mask is not None:
                label = label[mask.view(-1)]
    #print('label', label.shape, label)
    return label

def _flatten_aux_label(aux_label_pf, aux_mask_pf=None):
    if isinstance(aux_label_pf,(torch.LongTensor, torch.cuda.LongTensor)):
        aux_label_pf=(aux_label_pf.max(2)[1]).flatten().masked_select(aux_mask_pf).long()
    elif isinstance(aux_label_pf,(torch.FloatTensor, torch.cuda.FloatTensor)):
        aux_label_pf=aux_label_pf.flatten(end_dim=1)[aux_mask_pf, :].float()
    else:
        raise ValueError

    return aux_label_pf

def _flatten_preds(preds, mask=None, label_axis=1):
    if preds.ndim > 2:
        # assuming axis=1 corresponds to the classes
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    #print('preds', preds.shape, preds)
    return preds

def _trace_handler(prof):
    #print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    pass

def _aux_halder(aux_output, aux_label_pf, aux_mask_pf, aux_loss_func,
                num_aux_examples_pf,total_aux_correct_pf, loss, aux_loss_tot, dev,
                aux_label_counter=None, aux_scores=None ):

    #WARNING the mask is not valid if the tensor in question is the pf_vtx beacuse
    # in that case the invalid value is 0 and not -1
    if aux_mask_pf is None:
        aux_mask_pf = (aux_label_pf[:, :, 0] != -1).flatten()
    #print('\n aux_label_pf1\n', aux_label_pf.size(), aux_label_pf)

    aux_logits_pf=aux_output.flatten(end_dim=1).to(dev)[aux_mask_pf, :]
    aux_label_pf = _flatten_aux_label(aux_label_pf, aux_mask_pf)

    if num_aux_examples_pf == 0:
        num_aux_examples_pf = aux_label_pf.size(0)
    #print('\n aux_label_pf2\n', aux_label_pf.size(), aux_label_pf)
    #print('\n aux_mask_pf1\n', aux_mask_pf, aux_mask_pf.size())
    #print('\n num_aux_examples_pf\n', num_aux_examples_pf)
    #print('\n aux_logits_pf_clas\n', aux_logits_pf, aux_logits_pf.size())

    #HERE? scores
    if aux_label_counter is not None:
        aux_label_counter.update(aux_label_pf.cpu().numpy())
    if aux_scores is not None:
        aux_scores.append(torch.softmax(aux_logits_pf, dim=1).detach().cpu().numpy())

    aux_loss_pf = 0 if aux_loss_func is None else aux_loss_func(aux_logits_pf, aux_label_pf).to(dev).flatten()
    #print('\naux_loss\n', aux_loss_pf)
    if not isinstance(aux_loss_pf, int):
        try:
            loss = torch.cat((loss, aux_loss_pf), dim=0)
        except TypeError:
            loss = aux_loss_pf
        try:
            aux_loss_tot = torch.cat((aux_loss_tot, aux_loss_pf), dim=0)
        except TypeError:
            aux_loss_tot = aux_loss_pf
    #print('\nloss_sum\n', loss)

    if isinstance(aux_label_pf,(torch.LongTensor, torch.cuda.LongTensor)):
        _, aux_preds_pf = aux_logits_pf.max(1)
        aux_correct_pf = (aux_preds_pf == aux_label_pf).sum().item()
        total_aux_correct_pf += aux_correct_pf
    elif isinstance(aux_label_pf,(torch.FloatTensor, torch.cuda.FloatTensor)):
        #HERE sqrt? avg?
        aux_correct_pf = (aux_logits_pf - aux_label_pf).square().sum().item()

    return aux_label_pf, aux_mask_pf, loss, aux_loss_pf, aux_correct_pf, total_aux_correct_pf, num_aux_examples_pf, aux_label_counter, aux_scores

def train_classification(model, loss_func, aux_loss_func_clas, aux_loss_func_regr, aux_loss_func_bin, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    #HERE
    label_counter = Counter()
    aux_label_counter = Counter()
    total_loss = 0
    total_aux_loss = 0
    num_batches = 0
    total_correct = 0
    total_aux_correct_pf_clas = 0
    total_aux_correct_pf_regr = 0
    count = 0
    aux_count_pf = 0

    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], profile_memory=True, record_shapes=True,
                        schedule=torch.profiler.schedule( wait=1, warmup=1, active=5, repeat=2, skip_first=1), on_trace_ready=_trace_handler) as prof:
            for X, y, _ in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long()
                #print('\n\nlabels_size\n', label)

                if len([k for k in data_config.aux_label_names if 'pf_clas' in k]) > 0:
                    aux_label_pf_clas = torch.stack([y[k].float() for k in data_config.aux_label_names if 'pf_clas' in k]).permute(1,2,0).to(dev).long()# (batch_size, num_pf, num_labels_pf_clas)
                    #print('\n\ aux_label_pf_clas\n', aux_label_pf_clas, aux_label_pf_clas.size())

                else:
                    aux_label_pf_clas = None

                if len([k for k in data_config.aux_label_names if 'pf_regr' in k]) > 0:
                    aux_label_pf_regr = torch.stack([y[k].float() for k in data_config.aux_label_names if 'pf_regr' in k]).permute(1,2,0).to(dev).float()
                else:
                    aux_label_pf_regr = None

                if len([k for k in data_config.aux_label_names if 'pair_bin' in k]) > 0:
                    aux_label_pair_bin= torch.stack([y[k].float() for k in data_config.aux_label_names if 'pair_bin' in k]).to(dev).float().permute(1,2,3,0)
                    #print('\n\ aux_label_pair_bin\n','\n', aux_label_pair_bin, aux_label_pair_bin.size())
                else:
                    aux_label_pair_bin= None

                #print(data_config.aux_label_names)
                #print('\n\naux_labels_size1\n', aux_label_pf_clas.size(), aux_label_pf_clas)

                try:
                    label_mask = y[data_config.label_names[0] + '_mask'].bool()
                    aux_label_mask = y[data_config.aux_label_names[0] + '_mask'].bool()
                except KeyError:
                    label_mask = None
                    aux_label_mask = None
                label = _flatten_label(label, label_mask)


                num_examples = label.shape[0]
                label_counter.update(label.cpu().numpy())
                label = label.to(dev)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                    #HERE
                    model_output_tot = model(*inputs)

                    if isinstance(model_output_tot, tuple):
                        model_output = model_output_tot[0]
                        aux_output = model_output_tot[1]
                        aux_output_pair = model_output_tot[2]
                       # print('\n\aux_output\n', aux_output.size(), aux_output_pair.size())

                    logits = _flatten_preds(model_output, label_mask)
                    #print('\nlogits\n', logits.size(), logits)
                    loss = loss_func(logits, label)
                    #print('\n\n\nloss', loss.size(), loss)

                    #HERE
                    aux_mask_pf = None
                    num_aux_examples_pf = 0
                    aux_correct_pf_clas = 0
                    index_i = 0
                    index_f = 0
                    aux_loss = 0

                    if aux_label_pf_clas is not None:
                        index_f += aux_label_pf_clas.size(2)
                        _, aux_mask_pf, loss, aux_loss, aux_correct_pf_clas, \
                            total_aux_correct_pf_clas, num_aux_examples_pf,\
                            aux_label_counter, _ = \
                            _aux_halder(aux_output[:, :, index_i:index_f],
                                        aux_label_pf_clas[:, :aux_output.size(1), :],
                                        aux_mask_pf, aux_loss_func_clas,
                                        num_aux_examples_pf,total_aux_correct_pf_clas,
                                        loss, aux_loss, dev, aux_label_counter)
                        index_i += aux_label_pf_clas.size(2)

                    if aux_label_pf_regr is not None:
                        index_f += aux_label_pf_regr.size(2)
                        _, aux_mask_pf, loss, aux_loss, aux_correct_pf_regr, \
                            total_aux_correct_pf_regr,\
                            num_aux_examples_pf, _, _ = \
                            _aux_halder(aux_output[:, :, index_i:index_f],
                                        aux_label_pf_regr[:, :aux_output.size(1), :],
                                        aux_mask_pf, aux_loss_func_regr,
                                        num_aux_examples_pf,total_aux_correct_pf_regr,
                                        loss, aux_loss, dev)
                        index_i+=aux_label_pf_regr.size(2)

                    """if aux_label_pair_bin is not None:
                        #HERE flatten aux_output_pair
                        # usa la maschera dove in qualche modo devi sapere il numero di pf per ogni
                        # elemento di una batch. inoltre la maschera deve considrare anche quali
                        # pf sono unmatched
                        num_pf = (aux_label_pair_bin[:, 0, :, 0] == -2).nonzero(as_tuple=False)
                        x=torch.cat(( torch.where(num_pf[:-1, 0] != num_pf[1:, 0])), dim=0)
                        num_pf= torch.cat((num_pf[0, 1].unsqueeze(dim=0), num_pf[x+1, 1]))
                        print('\n num_pf', num_pf)
                        mask= aux_label_pair_bin[:,0,:,0] != -2
                        print('\n mask', mask)

                        aux_label_pair_bin=aux_label_pair_bin.flatten(end_dim=2)
                        print('\n\ aux_label_pair_bin\n','\n', aux_label_pair_bin, aux_label_pair_bin.size())


                        '''index_f += aux_label_pf_regr.size(2)
                        _, aux_mask_pf, loss, aux_loss, aux_correct_pf_regr, \
                            total_aux_correct_pf_regr,\
                            num_aux_examples_pf, _, _ = \
                            _aux_halder(aux_output[:, :, index_i:index_f],
                                        aux_label_pf_regr[:, :aux_output.size(1), :],
                                        aux_mask_pf, aux_loss_func_regr,
                                        num_aux_examples_pf,total_aux_correct_pf_regr,
                                        loss, aux_loss, dev)
                        index_i+=aux_label_pf_regr.size(2)'''
"""


                    aux_count_pf += num_aux_examples_pf

                loss = loss.mean()
                if grad_scaler is None:
                    loss.backward()
                    opt.step()
                else:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(opt)
                    grad_scaler.update()

                if scheduler and getattr(scheduler, '_update_per_step', False):
                    scheduler.step()

                _, preds = logits.max(1)
                loss = loss.item()

                if not isinstance(aux_loss, int):
                    aux_loss = aux_loss.mean()
                    aux_loss = aux_loss.item()

                total_aux_loss += aux_loss

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss
                total_correct += correct

                try:
                    aux_acc=aux_correct_pf_clas / num_aux_examples_pf
                    avg_aux_acc=total_aux_correct_pf_clas / aux_count_pf
                except ZeroDivisionError:
                    aux_acc=0
                    avg_aux_acc=0

                tq.set_postfix({
                    'Train epoch':epoch,
                    'Steps':steps_per_epoch,
                    'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count),
                    'AuxLoss': '%.5f' % aux_loss,
                    'AvgAuxLoss': '%.5f' % (total_aux_loss / num_batches),
                    'AuxAcc': '%.5f' % (aux_acc),
                    'AvgAuxAcc': '%.5f' % (avg_aux_acc)})

                if tb_helper:
                    tb_helper.write_scalars([
                        ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                        ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                        ])
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

                # send a signal to the profiler that the next iteration has started
                prof.step()


    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in %s (avg. speed %.1f entries/s)' % (count, time.strftime("%H:%M:%S", time.gmtime(time_diff)), count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))
    _logger.info('Train auxliliary class distribution: \n    %s', str(sorted(aux_label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification(model, test_loader, dev, epoch, for_training=True, loss_func=None,
                            aux_loss_func_clas=None, aux_loss_func_regr=None, aux_loss_func_bin=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix', 'save_labels'],
                            tb_helper=None, roc_prefix=None):
    model.eval()

    data_config = test_loader.dataset.config

    #HERE
    label_counter = Counter()
    aux_label_counter = Counter()
    total_loss = 0
    total_aux_loss = 0
    num_batches = 0
    total_correct = 0
    total_aux_correct_pf_clas = 0
    total_aux_correct_pf_regr = 0
    entry_count = 0
    count = 0
    aux_count_pf = 0
    scores = []
    aux_scores = []
    #HERE aux_labels to pass to metrics
    labels = defaultdict(list)
    aux_labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long()
                #print('\n\data_config.aux_label_names\n', data_config.aux_label_names)

                if len([k for k in data_config.aux_label_names if 'pf_clas' in k]) > 0:
                    aux_label_pf_clas = torch.stack([y[k].float() for k in data_config.aux_label_names if 'pf_clas' in k]).permute(1,2,0).to(dev).long()# (batch_size, num_pf, num_labels_pf_clas)
                else:
                    aux_label_pf_clas = None

                if len([k for k in data_config.aux_label_names if 'pf_regr' in k]) > 0:
                    aux_label_pf_regr = torch.stack([y[k].float() for k in data_config.aux_label_names if 'pf_regr' in k]).permute(1,2,0).to(dev).float()
                else:
                    aux_label_pf_regr = None

                if len([k for k in data_config.aux_label_names if 'pair_bin' in k]) > 0:
                    aux_label_pair_bin= torch.stack([y[k].float() for k in data_config.aux_label_names if 'pair_bin' in k]).permute(1,2,3,0).to(dev).long()
                else:
                    aux_label_pair_bin= None

                #print(data_config.aux_label_names)
                #print('\n\naux_labels_size1\n', aux_label_pf_clas.size(), aux_label_pf_clas)

                #HERE?
                entry_count += label.shape[0]

                try:
                    label_mask = y[data_config.label_names[0] + '_mask'].bool()
                    aux_label_mask = y[data_config.aux_label_names[0] + '_mask'].bool()
                except KeyError:
                    label_mask = None
                    aux_label_mask = None
                    #HERE put theat if the aux_label_mask is set
                    # the aux_mask is that and it is not calculated

                #HERE?
                if not for_training and label_mask is not None:
                    labels_counts.append(np.squeeze(label_mask.numpy().sum(axis=-1)))
                label = _flatten_label(label, label_mask)

                num_examples = label.shape[0]
                label_counter.update(label.cpu().numpy())
                label = label.to(dev)

                #HERE
                model_output_tot = model(*inputs)

                if isinstance(model_output_tot, tuple):
                    model_output = model_output_tot[0]
                    aux_output = model_output_tot[1]
                    aux_output_pair = model_output_tot[2]
                    #print('\n\aux_output\n', aux_output, aux_output_pair)

                logits = _flatten_preds(model_output, label_mask).float()
                scores.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                loss = 0 if loss_func is None else loss_func(logits, label)

                #HERE
                aux_mask_pf = None
                num_aux_examples_pf = 0
                aux_correct_pf_clas = 0
                index_i = 0
                index_f = 0
                aux_loss = 0
                if aux_label_pf_clas is not None:
                    index_f += aux_label_pf_clas.size(2)
                    aux_label_pf_clas_mask, aux_mask_pf,loss, aux_loss, aux_correct_pf_clas, \
                        total_aux_correct_pf_clas,\
                        num_aux_examples_pf, aux_label_counter, aux_scores = \
                        _aux_halder(aux_output[:, :, index_i:index_f],
                                    aux_label_pf_clas[:, :aux_output.size(1), :],
                                    aux_mask_pf, aux_loss_func_clas,
                                    num_aux_examples_pf,total_aux_correct_pf_clas,
                                    loss, aux_loss, dev, aux_label_counter, aux_scores)
                    index_i += aux_label_pf_clas.size(2)
                    aux_labels['aux_label_pf_clas'].append(aux_label_pf_clas_mask.cpu().numpy())

                if aux_label_pf_regr is not None:
                    index_f += aux_label_pf_regr.size(2)
                    _, aux_mask_pf,loss, aux_loss, aux_correct_pf_regr, \
                        total_aux_correct_pf_regr,\
                        num_aux_examples_pf, _, _ = \
                        _aux_halder(aux_output[:, :, index_i:index_f],
                                    aux_label_pf_regr[:, :aux_output.size(1), :],
                                    aux_mask_pf, aux_loss_func_regr,
                                    num_aux_examples_pf,total_aux_correct_pf_regr,
                                    loss, aux_loss, dev)
                    index_i += aux_label_pf_regr.size(2)


                aux_count_pf += num_aux_examples_pf

                for k, v in y.items():
                    if 'aux' not in k:
                        labels[k].append(_flatten_label(v, label_mask).cpu().numpy())
                        #print('v_label\n', v)

                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())

                _, preds = logits.max(1)

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()

                if not isinstance(loss, int):
                    loss = loss.mean()
                    loss = loss.item()

                if not isinstance(aux_loss, int):
                    aux_loss = aux_loss.mean()
                    aux_loss = aux_loss.item()

                total_aux_loss += aux_loss * num_aux_examples_pf

                total_loss += loss * num_examples
                total_correct += correct

                try:
                    aux_acc=aux_correct_pf_clas / num_aux_examples_pf
                    avg_aux_acc=total_aux_correct_pf_clas / aux_count_pf
                    avg_aux_loss = total_aux_loss / aux_count_pf
                except ZeroDivisionError:
                    aux_acc=0
                    avg_aux_acc=0
                    avg_aux_loss=0

                tq.set_postfix({
                    'Val epoch':epoch,
                    'Steps':steps_per_epoch,
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count),
                    'AuxLoss': '%.5f' % aux_loss,
                    'AvgAuxLoss': '%.5f' % (avg_aux_loss),
                    'AuxAcc': '%.5f' % (aux_acc),
                    'AvgAuxAcc': '%.5f' % (avg_aux_acc)})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

                #if num_batches % 500 == 0:


    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in %s (avg. speed %.1f entries/s)' % (count, time.strftime("%H:%M:%S", time.gmtime(time_diff)), count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))
    _logger.info('Train auxliliary class distribution: \n    %s', str(sorted(aux_label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    #print('labels', labels)
    #print('scores', scores)

    labels = {k: _concat(v) for k, v in labels.items()}
    #print('labels2', labels)
    #print('aux_labels', aux_labels)
    #print('aux_scores\n', aux_scores)

    try:
        aux_scores = np.concatenate(aux_scores)
        aux_labels = {k: _concat(v) for k, v in aux_labels.items()}
    except ValueError:
        aux_scores = None
        aux_labels = None
    #print('aux_labels2', aux_labels)
    #print('aux_scores\n', aux_scores)

    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores,
                        aux_labels, aux_scores, eval_metrics, epoch, roc_prefix)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_correct / count, total_loss / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert(count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, total_loss / count, scores, labels, observers


def save_labels_best_epoch(infile):
    with open(infile, 'rb') as in_f:
        label_file=np.load(in_f)
        y_true=label_file['y_true']
        y_score=label_file['y_score']

    with open(f'{infile.split("epoch")[0]}best_epoch{os.path.splitext(infile)[1]}', 'wb') as out_f:
        np.savez(out_f, y_true=y_true, y_score=y_score)


def evaluate_onnx(model_path, test_loader,
                  eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix', 'save_labels'],
                  epoch=-1, roc_prefix=None):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path)

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_correct = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with tqdm.tqdm(test_loader) as tq:
        for X, y, Z in tq:
            inputs = {k: v.cpu().numpy() for k, v in X.items()}
            label = y[data_config.label_names[0]].cpu().numpy()
            num_examples = label.shape[0]
            label_counter.update(label)
            score = sess.run([], inputs)[0]
            preds = score.argmax(1)

            scores.append(score)
            for k, v in y.items():
                labels[k].append(v.cpu().numpy())
            for k, v in Z.items():
                observers[k].append(v.cpu().numpy())

            correct = (preds == label).sum()
            total_correct += correct
            count += num_examples

            tq.set_postfix({
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in %s (avg. speed %.1f entries/s)' % (count, time.strftime("%H:%M:%S", time.gmtime(time_diff)), count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics, epoch=epoch, roc_prefix=roc_prefix)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
    observers = {k: _concat(v) for k, v in observers.items()}
    return total_correct / count, scores, labels, observers


def train_regression(model, loss_func, aux_loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].float()
            num_examples = label.shape[0]
            label = label.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                preds = model_output.squeeze()
                loss = loss_func(preds, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            loss = loss.item()

            num_batches += 1
            count += num_examples
            total_loss += loss
            e = preds - label
            abs_err = e.abs().sum().item()
            sum_abs_err += abs_err
            sqr_err = e.square().sum().item()
            sum_sqr_err += sqr_err

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'MSE': '%.5f' % (sqr_err / num_examples),
                'AvgMSE': '%.5f' % (sum_sqr_err / count),
                'MAE': '%.5f' % (abs_err / num_examples),
                'AvgMAE': '%.5f' % (sum_abs_err / count),
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in %s (avg. speed %.1f entries/s)' % (count, time.strftime("%H:%M:%S", time.gmtime(time_diff)), count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgMSE: %.5f, AvgMAE: %.5f' %
                 (total_loss / num_batches, sum_sqr_err / count, sum_abs_err / count))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            ("MAE/train (epoch)", sum_abs_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_regression(model, test_loader, dev, epoch, for_training=True, loss_func=None, aux_loss_func=None, steps_per_epoch=None,
                        eval_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                      'mean_gamma_deviance'],
                        tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].float()
                num_examples = label.shape[0]
                label = label.to(dev)
                model_output = model(*inputs)
                preds = model_output.squeeze().float()

                scores.append(preds.detach().cpu().numpy())
                for k, v in y.items():
                    labels[k].append(v.cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())

                loss = 0 if loss_func is None else loss_func(preds, label).item()

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples
                e = preds - label
                abs_err = e.abs().sum().item()
                sum_abs_err += abs_err
                sqr_err = e.square().sum().item()
                sum_sqr_err += sqr_err

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'MSE': '%.5f' % (sqr_err / num_examples),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    'MAE': '%.5f' % (abs_err / num_examples),
                    'AvgMAE': '%.5f' % (sum_abs_err / count),
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in %s (avg. speed %.1f entries/s)' % (count, time.strftime("%H:%M:%S", time.gmtime(time_diff)), count / time_diff))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
            ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_loss / count, None
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / count, None, scores, labels, observers


class TensorboardHelper(object):

    def __init__(self, tb_comment, tb_custom_fn):
        self.tb_comment = tb_comment
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(comment=self.tb_comment)
        _logger.info('Create Tensorboard summary writer with comment %s' % self.tb_comment)

        # initiate the batch state
        self.batch_train_count = 0

        # load custom function
        self.custom_fn = tb_custom_fn
        if self.custom_fn is not None:
            from weaver.utils.import_tools import import_module
            from functools import partial
            self.custom_fn = import_module(self.custom_fn, '_custom_fn')
            self.custom_fn = partial(self.custom_fn.get_tensorboard_custom_fn, tb_writer=self.writer)

    def __del__(self):
        self.writer.close()

    def write_scalars(self, write_info):
        for tag, scalar_value, global_step in write_info:
            self.writer.add_scalar(tag, scalar_value, global_step)
