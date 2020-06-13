import numpy as np
import awkward
import tqdm
import torch

from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger


def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label


def _flatten_preds(preds, mask=None, label_axis=1):
    if preds.ndim > 2:
        # assuming axis=1 corresponds to the classes
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    # print('preds', preds.shape, preds)
    return preds


def train(model, loss_func, opt, scheduler, train_loader, dev, use_amp=False):
    model.train()

    data_config = train_loader.dataset.config

    if use_amp:
        from apex import amp

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long()
            try:
                label_mask = y[data_config.label_names[0] + '_mask'].bool()
            except KeyError:
                label_mask = None
            label = _flatten_label(label, label_mask)
            num_examples = label.shape[0]
            label_counter.update(label.cpu().numpy())
            label = label.to(dev)
            opt.zero_grad()
            logits = model(*inputs)
            logits = _flatten_preds(logits, label_mask)
            loss = loss_func(logits, label)
            if use_amp:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            opt.step()

            _, preds = logits.max(1)
            loss = loss.item()

            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))
    scheduler.step()


def evaluate(model, test_loader, dev, for_training=True, loss_func=None, eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long()
                entry_count += label.shape[0]
                try:
                    label_mask = y[data_config.label_names[0] + '_mask'].bool()
                except KeyError:
                    label_mask = None
                if not for_training and label_mask is not None:
                    labels_counts.append(np.squeeze(label_mask.numpy().sum(axis=-1)))
                label = _flatten_label(label, label_mask)
                num_examples = label.shape[0]
                label_counter.update(label.cpu().numpy())
                label = label.to(dev)
                logits = model(*inputs)
                logits = _flatten_preds(logits, label_mask)

                scores.append(torch.softmax(logits, dim=1).cpu().detach().numpy())
                for k, v in y.items():
                    labels[k].append(_flatten_label(v, label_mask).cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())

                _, preds = logits.max(1)
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_correct / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = awkward.JaggedArray.fromcounts(labels_counts, scores)
                for k, v in labels.items():
                    labels[k] = awkward.JaggedArray.fromcounts(labels_counts, v)
            else:
                assert(count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers


def evaluate_onnx(model_path, test_loader, eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path)

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_correct = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    with tqdm.tqdm(test_loader) as tq:
        for X, y, Z in tq:
            inputs = {k:v.cpu().numpy() for k, v in X.items()}
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

    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))
    scores = np.concatenate(scores)
    labels = {k:_concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
    observers = {k:_concat(v) for k, v in observers.items()}
    return total_correct / count, scores, labels, observers
