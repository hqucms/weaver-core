#!/usr/bin/env python

import os
import shutil
import glob
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader
from importlib import import_module
import ast
from utils.logger import _logger, _configLogger
from utils.dataset import SimpleIterDataset

parser = argparse.ArgumentParser()
parser.add_argument('--regression-mode', action='store_true', default=False,
                    help='run in regression mode if this flag is set; otherwise run in classification mode')
parser.add_argument('-c', '--data-config', type=str, default='data/ak15_points_pf_sv_v0.yaml',
                    help='data config YAML file')
parser.add_argument('-i', '--data-train', nargs='*', default=[],
                    help='training files')
parser.add_argument('-t', '--data-test', nargs='*', default=[],
                    help='testing files')
parser.add_argument('--data-fraction', type=float, default=1,
                    help='fraction of events to load from each file; for training, the events are randomly selected for each epoch')
parser.add_argument('--file-fraction', type=float, default=1,
                    help='fraction of files to load; for training, the files are randomly selected for each epoch')
parser.add_argument('--fetch-by-files', action='store_true', default=False,
                    help='When enabled, will load all events from a small number (set by ``--fetch-step``) of files for each data fetching. '
                         'Otherwise (default), load a small fraction of events from all files each time, which helps reduce variations in the sample composition.')
parser.add_argument('--fetch-step', type=float, default=0.01,
                    help='fraction of events to load each time from every file (when ``--fetch-by-files`` is disabled); '
                         'Or: number of files to load each time (when ``--fetch-by-files`` is enabled). Shuffling & sampling is done within these events, so set a large enough value.')
parser.add_argument('--train-val-split', type=float, default=0.8,
                    help='training/validation split fraction')
parser.add_argument('--demo', action='store_true', default=False,
                    help='quickly test the setup by running over only a small number of events')
parser.add_argument('--lr-finder', type=str, default=None,
                    help='run learning rate finder instead of the actual training; format: ``start_lr, end_lr, num_iters``')
parser.add_argument('-n', '--network-config', type=str, default='networks/particle_net_pfcand_sv.py',
                    help='network architecture configuration file; the path must be relative to the current dir')
parser.add_argument('-o', '--network-option', nargs=2, action='append', default=[],
                    help='options to pass to the model class constructor, e.g., `--network-option use_counts False`')
parser.add_argument('-m', '--model-prefix', type=str, default='models/{auto}/network',
                    help='path to save or load the model; for training, this will be used as a prefix, so model snapshots '
                         'will saved to `{model_prefix}_epoch-%d_state.pt` after each epoch, and the one with the best '
                         'validation metric to `{model_prefix}_best_epoch_state.pt`; for testing, this should be the full path '
                         'including the suffix, otherwise the one with the best validation metric will be used; '
                         'for training, `{auto}` can be used as part of the path to auto-generate a name, '
                         'based on the timestamp and network configuration')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='number of epochs')
parser.add_argument('--optimizer', type=str, default='ranger', choices=['adam', 'adamW', 'ranger'],  # TODO: add more
                    help='optimizer for the training')
parser.add_argument('--load-epoch', type=int, default=None,
                    help='used to resume interrupted training, load model and optimizer state saved in the `epoch-%d_state.pt` and `epoch-%d_optimizer.pt` files')
parser.add_argument('--start-lr', type=float, default=5e-3,
                    help='start learning rate')
parser.add_argument('--lr-steps', type=str, default='10,20',
                    help='steps to reduce the lr; currently only used when setting `--optimizer` to adam')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--use-amp', action='store_true', default=False,
                    help='use mixed precision training (fp16); NOT WORKING YET')
parser.add_argument('--gpus', type=str, default='0',
                    help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with this numbers')
parser.add_argument('--predict', action='store_true', default=False,
                    help='run prediction instead of training')
parser.add_argument('--predict-output', type=str,
                    help='path to save the prediction output, support `.root` and `.awkd` format')
parser.add_argument('--export-onnx', type=str, default=None,
                    help='export the PyTorch model to ONNX model and save it at the given path (path must ends w/ .onnx); '
                         'needs to set `--data-config`, `--network-config`, and `--model-prefix` (requires the full model path)')
parser.add_argument('--io-test', action='store_true', default=False,
                    help='test throughput of the dataloader')
parser.add_argument('--copy-inputs', action='store_true', default=False,
                    help='copy input files to the current dir (can help to speed up dataloading when running over remote files, e.g., from EOS)')
parser.add_argument('--log', type=str, default='',
                    help='path to the log file; `{auto}` can be used as part of the path to auto-generate a name, based on the timestamp and network configuration')
parser.add_argument('--print', action='store_true', default=False,
                    help='do not run training/prediction but only print model information, e.g., FLOPs and number of parameters of a model')


def train_load(args):
    """
    Loads the training data.
    :param args:
    :return: train_loader, val_loader, data_config, train_inputs
    """
    filelist = sorted(sum([glob.glob(f) for f in args.data_train], []))
    if args.copy_inputs:
        import tempfile
        tmpdir = tempfile.mkdtemp()
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        new_filelist = []
        for src in filelist:
            dest = os.path.join(tmpdir, src.lstrip('/'))
            if not os.path.exists(os.path.dirname(dest)):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
            _logger.info('Copied file %s to %s' % (src, dest))
            new_filelist.append(dest)
        filelist = new_filelist

    # np.random.seed(1)
    np.random.shuffle(filelist)
    if args.demo:
        filelist = filelist[:20]
        _logger.info(filelist)
        args.data_fraction = 0.1
        args.fetch_step = 0.002
    num_workers = min(args.num_workers, int(len(filelist) * args.file_fraction))
    train_data = SimpleIterDataset(filelist, args.data_config, for_training=True,
                                   load_range_and_fraction=((0, args.train_val_split), args.data_fraction),
                                   file_fraction=args.file_fraction, fetch_by_files=args.fetch_by_files,
                                   fetch_step=args.fetch_step)
    val_data = SimpleIterDataset(filelist, args.data_config, for_training=True,
                                 load_range_and_fraction=((args.train_val_split, 1), args.data_fraction),
                                 file_fraction=args.file_fraction, fetch_by_files=args.fetch_by_files,
                                 fetch_step=args.fetch_step)
    train_loader = DataLoader(train_data, num_workers=num_workers, batch_size=args.batch_size, drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data, num_workers=num_workers, batch_size=args.batch_size, drop_last=True,
                            pin_memory=True)
    data_config = train_data.config
    train_input_names = train_data.config.input_names
    train_label_names = train_data.config.label_names

    return train_loader, val_loader, data_config, train_input_names, train_label_names


def test_load(args):
    """
    Loads the test data.
    :param args:
    :return: test_loader, data_config
    """
    filelist = sorted(sum([glob.glob(f) for f in args.data_test], []))
    num_workers = min(args.num_workers, len(filelist))
    test_data = SimpleIterDataset(filelist, args.data_config, for_training=False,
                                  load_range_and_fraction=((0, 1), args.data_fraction),
                                  fetch_by_files=True, fetch_step=1)
    test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=args.batch_size, drop_last=False,
                             pin_memory=True)
    data_config = test_data.config
    return test_loader, data_config


def onnx(args, model, data_config, model_info):
    """
    Saving model as ONNX.
    :param args:
    :param model:
    :param data_config:
    :param model_info:
    :return:
    """
    assert (args.export_onnx.endswith('.onnx'))
    model_path = args.model_prefix
    _logger.info('Exporting model %s to ONNX' % model_path)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.cpu()
    model.eval()

    os.makedirs(os.path.dirname(args.export_onnx), exist_ok=True)
    inputs = tuple(
        torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])
    torch.onnx.export(model, inputs, args.export_onnx,
                      input_names=model_info['input_names'],
                      output_names=model_info['output_names'],
                      dynamic_axes=model_info.get('dynamic_axes', None),
                      opset_version=11)
    _logger.info('ONNX model saved to %s', args.export_onnx)

    preprocessing_json = os.path.join(os.path.dirname(args.export_onnx), 'preprocess.json')
    data_config.export_json(preprocessing_json)
    _logger.info('Preprocessing parameters saved to %s', preprocessing_json)


def flops(model, model_info):
    """
    Count FLOPs and params.
    :param args:
    :param model:
    :param model_info:
    :return:
    """
    from utils.flops_counter import get_model_complexity_info
    import copy

    model = copy.deepcopy(model).cpu()
    model.eval()

    inputs = tuple(
        torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])

    macs, params = get_model_complexity_info(model, inputs, as_strings=True, print_per_layer_stat=True, verbose=True)
    _logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    _logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))


def optim(args, model):
    """
    Optimizer and scheduler. Could try CosineAnnealing
    :param args:
    :param model:
    :return:
    """
    scheduler = None
    if args.optimizer == 'ranger':
        from utils.nn.optimizer.ranger import Ranger
        opt = Ranger(model.parameters(), lr=args.start_lr)
        if args.lr_finder is None:
            lr_decay_epochs = max(1, int(args.num_epochs * 0.3))
            lr_decay_rate = 0.01 ** (1. / lr_decay_epochs)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(
                range(args.num_epochs - lr_decay_epochs, args.num_epochs)), gamma=lr_decay_rate)
    else:
        if args.optimizer == 'adam':
            opt = torch.optim.Adam(model.parameters(), lr=args.start_lr)
        elif args.optimizer == 'adamW':
            opt = torch.optim.AdamW(model.parameters(), lr=args.start_lr)
        if args.lr_finder is None:
            lr_steps = [int(x) for x in args.lr_steps.split(',')]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_steps, gamma=0.1)
    return opt, scheduler


def model_setup(args, data_config):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config.replace('.py', '').replace('/', '.'))
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    _logger.info('Network options: %s' % str(network_options))
    if args.export_onnx:
        network_options['for_inference'] = True
    if args.use_amp:
        network_options['use_amp'] = True
    model, model_info = network_module.get_model(data_config, **network_options)
    # _logger.info(model)
    flops(model, model_info)
    return model, model_info, network_module, network_options


def iotest(args, data_loader):
    """
    Io test
    :param args:
    :param data_loader:
    :return:
    """
    from tqdm.auto import tqdm
    from collections import defaultdict
    from utils.data.tools import _concat
    _logger.info('Start running IO test')
    monitor_info = defaultdict(list)

    for X, y, Z in tqdm(data_loader):
        for k, v in Z.items():
            monitor_info[k].append(v.cpu().numpy())
    monitor_info = {k: _concat(v) for k, v in monitor_info.items()}
    if monitor_info:
        monitor_output_path = 'weaver_monitor_info.pkl'
        import pickle
        with open(monitor_output_path, 'wb') as f:
            pickle.dump(monitor_info, f)
        _logger.info('Monitor info written to %s' % monitor_output_path)


def save_root(data_config, scores, labels, observers):
    """
    Saves as .root
    :param data_config:
    :param scores:
    :param labels
    :param observers
    :return:
    """
    from utils.data.fileio import _write_root
    output = {}
    if args.regression_mode:
        output[data_config.label_names[0]] = labels[data_config.label_names[0]]
        output['output'] = scores
    else:
        for idx, label_name in enumerate(data_config.label_value):
            output[label_name] = (labels[data_config.label_names[0]] == idx)
            output['score_' + label_name] = scores[:, idx]
    for k, v in labels.items():
        if k == data_config.label_names[0]:
            continue
        if v.ndim > 1:
            _logger.warning('Ignoring %s, not a 1d array.', k)
            continue
        output[k] = v
    for k, v in observers.items():
        if v.ndim > 1:
            _logger.warning('Ignoring %s, not a 1d array.', k)
            continue
        output[k] = v
    _write_root(args.predict_output, output)


def save_awk(scores, labels, observers):
    """
    Saves as .awkd
    :param scores:
    :param labels:
    :param observers:
    :return:
    """
    from utils.data.tools import awkward
    output = {'scores': scores}
    output.update(labels)
    output.update(observers)

    name_remap = {}
    arraynames = list(output)
    for i in range(len(arraynames)):
        for j in range(i + 1, len(arraynames)):
            if arraynames[i].startswith(arraynames[j]):
                name_remap[arraynames[j]] = '%s_%d' % (arraynames[j], len(name_remap))
            if arraynames[j].startswith(arraynames[i]):
                name_remap[arraynames[i]] = '%s_%d' % (arraynames[i], len(name_remap))
    _logger.info('Renamed the following variables in the output file: %s', str(name_remap))
    output = {name_remap[k] if k in name_remap else k: v for k, v in output.items()}

    awkward.save(args.predict_output, output, mode='w')


def main(args):
    _logger.info(args)

    if args.file_fraction < 1:
        _logger.warning('Use of `file-fraction` is not recommended in general -- prefer using `data-fraction` instead.')

    # classification/regression mode
    if args.regression_mode:
        _logger.info('Running in regression mode')
        from utils.nn.tools import train_regression as train
        from utils.nn.tools import evaluate_regression as evaluate
    else:
        _logger.info('Running in classification mode')
        from utils.nn.tools import train_classification as train
        from utils.nn.tools import evaluate_classification as evaluate

    # training/testing mode
    training_mode = not args.predict

    # device
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(',')]
        dev = torch.device(gpus[0])
    else:
        gpus = None
        dev = torch.device('cpu')

    # load data
    if training_mode:
        train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(args)
    else:
        test_loader, data_config = test_load(args)

    if args.io_test:
        data_loader = train_loader if training_mode else test_loader
        iotest(args, data_loader)
        return

    model, model_info, network_module, network_options = model_setup(args, data_config)

    if args.print:
        return

    # export to ONNX
    if args.export_onnx:
        onnx(args, model, data_config, model_info)
        return

    # note: we should always save/load the state_dict of the original model, not the one wrapped by nn.DataParallel
    # so we do not convert it to nn.DataParallel now
    orig_model = model

    if training_mode:
        model = orig_model.to(dev)
        # loss function
        try:
            loss_func = network_module.get_loss(data_config, **network_options)
            _logger.info('Using loss function %s with options %s' % (loss_func, network_options))
        except AttributeError:
            loss_func = torch.nn.CrossEntropyLoss()
            _logger.warning('Loss function not defined in %s. Will use `torch.nn.CrossEntropyLoss()` by default.',
                            args.network_config)

        # optimizer & learning rate
        opt, scheduler = optim(args, model)

        # load previous training and resume if `--load-epoch` is set
        if args.load_epoch is not None:
            _logger.info('Resume training from epoch %d' % args.load_epoch)
            model_state = torch.load(args.model_prefix + '_epoch-%d_state.pt' % args.load_epoch, map_location=dev)
            model.load_state_dict(model_state)
            opt_state = torch.load(args.model_prefix + '_epoch-%d_optimizer.pt' % args.load_epoch, map_location=dev)
            opt.load_state_dict(opt_state)

        # multi-gpu
        if gpus is not None and len(gpus) > 1:
            # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
            model = torch.nn.DataParallel(model, device_ids=gpus)
        model = model.to(dev)

        # lr finder: keep it after all other setups
        if args.lr_finder is not None:
            start_lr, end_lr, num_iter = args.lr_finder.replace(' ', '').split(',')
            from utils.lr_finder import LRFinder
            lr_finder = LRFinder(model, opt, loss_func, device=dev, input_names=train_input_names,
                                 label_names=train_label_names)
            lr_finder.range_test(train_loader, start_lr=float(start_lr), end_lr=float(end_lr), num_iter=int(num_iter))
            lr_finder.plot(output='lr_finder.png')  # to inspect the loss-learning rate graph
            return

        if args.use_amp:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
        else:
            scaler = None

        # training loop
        best_valid_metric = np.inf if args.regression_mode else 0
        for epoch in range(args.num_epochs):
            if args.load_epoch is not None:
                if epoch <= args.load_epoch:
                    continue
            print('-' * 50)
            _logger.info('Epoch #%d training' % epoch)
            train(model, loss_func, opt, scheduler, train_loader, dev, grad_scaler=scaler)
            if args.model_prefix:
                dirname = os.path.dirname(args.model_prefix)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_dict, args.model_prefix + '_epoch-%d_state.pt' % epoch)
                torch.save(opt.state_dict(), args.model_prefix + '_epoch-%d_optimizer.pt' % epoch)

            _logger.info('Epoch #%d validating' % epoch)
            valid_metric = evaluate(model, val_loader, dev, loss_func=loss_func)
            is_best_epoch = (
                valid_metric < best_valid_metric) if args.regression_mode else(
                valid_metric > best_valid_metric)
            if is_best_epoch:
                best_valid_metric = valid_metric
                if args.model_prefix:
                    shutil.copy2(args.model_prefix + '_epoch-%d_state.pt' %
                                 epoch, args.model_prefix + '_best_epoch_state.pt')
                    torch.save(model, args.model_prefix + '_best_epoch_full.pt')
            _logger.info('Epoch #%d: Current validation metric: %.5f (best: %.5f)' %
                         (epoch, valid_metric, best_valid_metric))

    if args.data_test:
        model = orig_model.to(dev)

        if training_mode:
            del train_loader, val_loader
            test_loader, data_config = test_load(args)
        # run prediction
        if args.model_prefix.endswith('.onnx'):
            _logger.info('Loading model %s for eval' % args.model_prefix)
            from utils.nn.tools import evaluate_onnx
            test_metric, scores, labels, observers = evaluate_onnx(args.model_prefix, test_loader)
        else:
            model_path = args.model_prefix if args.model_prefix.endswith(
                '.pt') else args.model_prefix + '_best_epoch_state.pt'
            _logger.info('Loading model %s for eval' % model_path)
            model.load_state_dict(torch.load(model_path, map_location=dev))
            if gpus is not None and len(gpus) > 1:
                model = torch.nn.DataParallel(model, device_ids=gpus)
            model = model.to(dev)
            test_metric, scores, labels, observers = evaluate(model, test_loader, dev, for_training=False)
        _logger.info('Test metric %.5f' % test_metric)

        if args.predict_output:
            if '/' not in args.predict_output:
                args.predict_output = os.path.join(
                    os.path.dirname(args.model_prefix),
                    'predict_output', args.predict_output)
            os.makedirs(os.path.dirname(args.predict_output), exist_ok=True)
            if args.predict_output.endswith('.root'):
                save_root(data_config, scores, labels, observers)
            else:
                save_awk(scores, labels, observers)
            _logger.info('Written output to %s' % args.predict_output)


if __name__ == '__main__':
    args = parser.parse_args()
    if '{auto}' in args.model_prefix or '{auto}' in args.log:
        import hashlib
        import time
        model_name = time.strftime('%Y%m%d-%H%M%S') + "_" + os.path.basename(args.network_config).replace('.py', '')
        if len(args.network_option):
            model_name = model_name + "_" + hashlib.md5(str(args.network_option).encode('utf-8')).hexdigest()
        model_name += '_{optim}_lr{lr}_batch{batch}'.format(lr=args.start_lr,
                                                            optim=args.optimizer, batch=args.batch_size)
        args._auto_model_name = model_name
        args.model_prefix = args.model_prefix.replace('{auto}', model_name)
        args.log = args.log.replace('{auto}', model_name)
        print('Using auto-generated model prefix %s' % args.model_prefix)
    _configLogger('weaver', filename=args.log)
    main(args)
