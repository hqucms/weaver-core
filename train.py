#!/usr/bin/env python

import os
import ast
import sys
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch

from torch.utils.data import DataLoader

#sys.path.append(os.path.join("..", ""))
from weaver.utils.logger import _logger, _configLogger
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module
from weaver.utils.nn.tools import save_labels_best_epoch


'''orig_stdout = sys.stdout
f = open('out_log.txt', 'w')
sys.stdout = f'''

parser = argparse.ArgumentParser()
parser.add_argument('--regression-mode', action='store_true', default=False,
                    help='run in regression mode if this flag is set; otherwise run in classification mode')
parser.add_argument('-c', '--data-config', type=str,
                    help='data config YAML file')
parser.add_argument('--extra-selection', type=str, default=None,
                    help='Additional selection requirement, will modify `selection` to `(selection) & (extra)` on-the-fly')
parser.add_argument('--extra-test-selection', type=str, default=None,
                    help='Additional test-time selection requirement, will modify `test_time_selection` to `(test_time_selection) & (extra)` on-the-fly')
parser.add_argument('-i', '--data-train', nargs='*', default=[],
                    help='training files; supported syntax:'
                         ' (a) plain list, `--data-train /path/to/a/* /path/to/b/*`;'
                         ' (b) (named) groups [Recommended], `--data-train a:/path/to/a/* b:/path/to/b/*`,'
                         ' the file splitting (for each dataloader worker) will be performed per group,'
                         ' and then mixed together, to ensure a uniform mixing from all groups for each worker.'
                    )
parser.add_argument('-l', '--data-val', nargs='*', default=[],
                    help='validation files; when not set, will use training files and split by `--train-val-split`')
parser.add_argument('-t', '--data-test', nargs='*', default=[],
                    help='testing files; supported syntax:'
                         ' (a) plain list, `--data-test /path/to/a/* /path/to/b/*`;'
                         ' (b) keyword-based, `--data-test a:/path/to/a/* b:/path/to/b/*`, will produce output_a, output_b;'
                         ' (c) split output per N input files, `--data-test a%10:/path/to/a/*`, will split per 10 input files')
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
parser.add_argument('--in-memory', action='store_true', default=False,
                    help='load the whole dataset (and perform the preprocessing) only once and keep it in memory for the entire run')
parser.add_argument('--train-val-split', type=float, default=0.8,
                    help='training/validation split fraction')
parser.add_argument('--no-remake-weights', action='store_true', default=False,
                    help='do not remake weights for sampling (reweighting), use existing ones in the previous auto-generated data config YAML file')
parser.add_argument('--demo', action='store_true', default=False,
                    help='quickly test the setup by running over only a small number of events')
parser.add_argument('--lr-finder', type=str, default=None,
                    help='run learning rate finder instead of the actual training; format: ``start_lr, end_lr, num_iters``')
parser.add_argument('--tensorboard', type=str, default=None,
                    help='create a tensorboard summary writer with the given comment')
parser.add_argument('--tensorboard-custom-fn', type=str, default=None,
                    help='the path of the python script containing a user-specified function `get_tensorboard_custom_fn`, '
                         'to display custom information per mini-batch or per epoch, during the training, validation or test.')
parser.add_argument('-n', '--network-config', type=str,
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
parser.add_argument('--load-model-weights', type=str, default=None,
                    help='initialize model with pre-trained weights')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='number of epochs')
parser.add_argument('--steps-per-epoch', type=int, default=None,
                    help='number of steps (iterations) per epochs; '
                         'if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples')
parser.add_argument('--steps-per-epoch-val', type=int, default=None,
                    help='number of steps (iterations) per epochs for validation; '
                         'if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples')
parser.add_argument('--samples-per-epoch', type=int, default=None,
                    help='number of samples per epochs; '
                         'if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples')
parser.add_argument('--samples-per-epoch-val', type=int, default=None,
                    help='number of samples per epochs for validation; '
                         'if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples')
parser.add_argument('--optimizer', type=str, default='ranger', choices=['adam', 'adamW', 'radam', 'ranger'],  # TODO: add more
                    help='optimizer for the training')
parser.add_argument('--optimizer-option', nargs=2, action='append', default=[],
                    help='options to pass to the optimizer class constructor, e.g., `--optimizer-option weight_decay 1e-4`')
parser.add_argument('--lr-scheduler', type=str, default='flat+decay',
                    choices=['none', 'steps', 'flat+decay', 'flat+linear', 'flat+cos', 'one-cycle'],
                    help='learning rate scheduler')
parser.add_argument('--warmup-steps', type=int, default=0,
                    help='number of warm-up steps, only valid for `flat+linear` and `flat+cos` lr schedulers')
parser.add_argument('--load-epoch', type=int, default=None,
                    help='used to resume interrupted training, load model and optimizer state saved in the `epoch-%d_state.pt` and `epoch-%d_optimizer.pt` files')
parser.add_argument('--start-lr', type=float, default=5e-3,
                    help='start learning rate')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--use-amp', action='store_true', default=False,
                    help='use mixed precision training (fp16)')
parser.add_argument('--gpus', type=str, default='0',
                    help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`')
parser.add_argument('--predict-gpus', type=str, default=None,
                    help='device for the testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`; if not set, use the same as `--gpus`')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with this numbers')
parser.add_argument('--predict', action='store_true', default=False,
                    help='run prediction instead of training')
parser.add_argument('--predict-output', type=str,
                    help='path to save the prediction output, support `.root` and `.parquet` format')
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
parser.add_argument('--profile', action='store_true', default=False,
                    help='run the profiler')
parser.add_argument('--backend', type=str, choices=['gloo', 'nccl', 'mpi'], default=None,
                    help='backend for distributed training')
parser.add_argument('--cross-validation', type=str, default=None,
                    help='enable k-fold cross validation; input format: `variable_name%k`')
parser.add_argument('--val', action='store_true', default=False,
                    help='perform only validation on the model state given in `--model-prefix`')
parser.add_argument('--train', action='store_true', default=False,
                    help='perform only training on the model state given in `--model-prefix`')
parser.add_argument('--test', action='store_true', default=False,
                    help='test the model on the `--data-test` sample')
parser.add_argument('--force-lr', action='store_true', default=False,
                    help='force the lr to be `--start-lr`')
parser.add_argument('--no-aux-epoch', type=float, default=1e9,
                    help='if epoch >= `--no-aux-epoch` do not consider auxiliary loss when training'
                    'unlsess `--aux-saturation` is set, in which case the auxiliary loss is saturated to the value'
                    'it has at `--no-aux-epoch`')
parser.add_argument('--aux-saturation', action='store_true', default=False,
                    help='if set, the auxiliary loss is saturated to the value it has at `--no-aux-epoch`')
parser.add_argument('--val-epochs', type=str, default='-1',
                    help='epochs on which the validation is performed'
                    'if not provided perform on every epoch'
                    'separate the epochs with `,` in order to indicate them one by one'
                    'separate the epochs with `:` in order to indicate an interval'
                    'if set to `best` the validation is performed on the best epoch')
parser.add_argument('--test-epochs', type=str, default='',
                    help='epochs on which the validation is performed'
                    'if not provided same as `--val-epochs`'
                    'if set to `-1` perform on all the epochs'
                    'separate the epochs with `,` in order to indicate them one by one'
                    'separate the epochs with `:` in order to indicate an interval'
                    'if set to `best+last` the validation is performed on the best and last epochs'
                    'if set to `best` the validation is performed on the best epoch')
parser.add_argument('--epoch-division', type=int, default=5,
                    help='number of printouts during an epoch')

def to_filelist(args, mode='train'):
    if mode == 'train':
        flist = args.data_train
    elif mode == 'val':
        flist = args.data_val
    else:
        raise NotImplementedError('Invalid mode %s' % mode)

    # keyword-based: 'a:/path/to/a b:/path/to/b'
    file_dict = {}
    for f in flist:
        if ':' in f:
            name, fp = f.split(':')
        else:
            name, fp = '_', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    if args.local_rank is not None:
        if mode == 'train':
            local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[args.local_rank::local_world_size]
                assert(len(new_files) > 0)
                np.random.shuffle(new_files)
                new_file_dict[name] = new_files
            file_dict = new_file_dict

    if args.copy_inputs:
        import tempfile
        tmpdir = tempfile.mkdtemp()
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        new_file_dict = {name: [] for name in file_dict}
        for name, files in file_dict.items():
            for src in files:
                dest = os.path.join(tmpdir, src.lstrip('/'))
                if not os.path.exists(os.path.dirname(dest)):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                _logger.info('Copied file %s to %s' % (src, dest))
                new_file_dict[name].append(dest)
            if len(files) != len(new_file_dict[name]):
                _logger.error('Only %d/%d files copied for %s file group %s',
                              len(new_file_dict[name]), len(files), mode, name)
        file_dict = new_file_dict

    filelist = sum(file_dict.values(), [])
    assert(len(filelist) == len(set(filelist)))
    return file_dict, filelist


def train_load(args):
    """
    Loads the training data.
    :param args:
    :return: train_loader, val_loader, data_config, train_inputs
    """

    train_file_dict, train_files = to_filelist(args, 'train')
    if args.data_val:
        val_file_dict, val_files = to_filelist(args, 'val')
        train_range = val_range = (0, 1)
    else:
        val_file_dict, val_files = train_file_dict, train_files
        train_range = (0, args.train_val_split)
        val_range = (args.train_val_split, 1)

    _logger.info('Using %d files for training, range: %s' % (len(train_files), str(train_range)))
    _logger.info('Using %d files for validation, range: %s' % (len(val_files), str(val_range)))

    if args.demo:
        train_files = train_files[:20]
        val_files = val_files[:20]
        train_file_dict = {'_': train_files}
        val_file_dict = {'_': val_files}
        _logger.info(train_files)
        _logger.info(val_files)
        args.data_fraction = 0.1
        args.fetch_step = 0.002

    if args.in_memory and (args.steps_per_epoch is None or args.steps_per_epoch_val is None):
        raise RuntimeError('Must set --steps-per-epoch when using --in-memory!')

    data_config = None
    train_input_names = None
    train_label_names = None
    train_aux_label_names = None

    if args.train:
        train_data = SimpleIterDataset(train_file_dict, args.data_config, for_training=True,
                                    extra_selection=args.extra_selection,
                                    remake_weights=not args.no_remake_weights,
                                    load_range_and_fraction=(train_range, args.data_fraction),
                                    file_fraction=args.file_fraction,
                                    fetch_by_files=args.fetch_by_files,
                                    fetch_step=args.fetch_step,
                                    infinity_mode=args.steps_per_epoch is not None,
                                    in_memory=args.in_memory,
                                    name='train' + ('' if args.local_rank is None else '_rank%d' % args.local_rank))
        train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, pin_memory=True,
                                num_workers=min(args.num_workers, int(len(train_files) * args.file_fraction)),
                                persistent_workers=args.num_workers > 0 and args.steps_per_epoch is not None)

        data_config = train_data.config
        train_input_names = train_data.config.input_names
        train_label_names = train_data.config.label_names
        train_aux_label_names = train_data.config.aux_label_names
    else:
        train_data = None
        train_loader = None

    if args.val:
        val_data = SimpleIterDataset(val_file_dict, args.data_config, for_training=True,
                                    extra_selection=args.extra_selection,
                                    load_range_and_fraction=(val_range, args.data_fraction),
                                    file_fraction=args.file_fraction,
                                    fetch_by_files=args.fetch_by_files,
                                    fetch_step=args.fetch_step,
                                    infinity_mode=args.steps_per_epoch_val is not None,
                                    in_memory=args.in_memory,
                                    name='val' + ('' if args.local_rank is None else '_rank%d' % args.local_rank))
        val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, pin_memory=True,
                                num_workers=min(args.num_workers, int(len(val_files) * args.file_fraction)),
                                persistent_workers=args.num_workers > 0 and args.steps_per_epoch_val is not None)
        if data_config is None: data_config = val_data.config
    else:
        val_data = None
        val_loader = None

    return train_loader, val_loader, data_config, train_input_names, train_label_names, train_aux_label_names


def test_load(args):
    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """
    # keyword-based --data-test: 'a:/path/to/a b:/path/to/b'
    # split --data-test: 'a%10:/path/to/a/*'
    file_dict = {}
    split_dict = {}
    for f in args.data_test:
        if ':' in f:
            name, fp = f.split(':')
            if '%' in name:
                name, split = name.split('%')
                split_dict[name] = int(split)
        else:
            name, fp = '', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f'{name}_{i}'] = files[i * split:(i + 1) * split]

    def get_test_loader(name):
        filelist = file_dict[name]
        _logger.info('Running on test file group %s with %d files:\n...%s', name, len(filelist), '\n...'.join(filelist))
        num_workers = min(args.num_workers, len(filelist))
        test_data = SimpleIterDataset({name: filelist}, args.data_config, for_training=False,
                                      extra_selection=args.extra_test_selection,
                                      load_range_and_fraction=((0, 1), args.data_fraction),
                                      fetch_by_files=True, fetch_step=1,
                                      name='test_' + name)
        test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=args.batch_size, drop_last=False,
                                 pin_memory=True)
        return test_loader

    test_loaders = {name: functools.partial(get_test_loader, name) for name in file_dict}
    data_config = SimpleIterDataset({}, args.data_config, for_training=False).config
    return test_loaders, data_config


def onnx(args):
    """
    Saving model as ONNX.
    :param args:
    :return:
    """
    assert (args.export_onnx.endswith('.onnx'))
    model_path = args.model_prefix
    _logger.info('Exporting model %s to ONNX' % model_path)

    from weaver.utils.dataset import DataConfig
    data_config = DataConfig.load(args.data_config, load_observers=False, load_reweight_info=False)
    model, model_info, _ = model_setup(args, data_config, torch.device('cpu'))
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
                      opset_version=13)
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
    from weaver.utils.flops_counter import get_model_complexity_info
    import copy

    model = copy.deepcopy(model).cpu()
    model.eval()

    inputs = tuple(
        torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])

    macs, params = get_model_complexity_info(model, inputs, as_strings=True, print_per_layer_stat=True, verbose=True)
    _logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    _logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))


def profile(args, model, model_info, device):
    """
    Profile.
    :param model:
    :param model_info:
    :return:
    """
    import copy
    from torch.profiler import profile, record_function, ProfilerActivity

    model = copy.deepcopy(model)
    model = model.to(device)
    model.eval()

    inputs = tuple(
        torch.ones((args.batch_size,) + model_info['input_shapes'][k][1:],
                   dtype=torch.float32).to(device) for k in model_info['input_names'])
    for x in inputs:
        print(x.shape, x.device)

    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
        print(output)
        p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=2),
        on_trace_ready=trace_handler
    ) as p:
        for idx in range(100):
            model(*inputs)
            p.step()


def optim(args, model, device):
    """
    Optimizer and scheduler.
    :param args:
    :param model:
    :return:
    """
    optimizer_options = {k: ast.literal_eval(v) for k, v in args.optimizer_option}
    _logger.info('Optimizer options: %s' % str(optimizer_options))

    names_lr_mult = []
    if 'weight_decay' in optimizer_options or 'lr_mult' in optimizer_options:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py#L31
        import re
        decay, no_decay = {}, {}
        names_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or (
                    hasattr(model, 'no_weight_decay') and name in model.no_weight_decay()):
                no_decay[name] = param
                names_no_decay.append(name)
            else:
                decay[name] = param

        decay_1x, no_decay_1x = [], []
        decay_mult, no_decay_mult = [], []
        mult_factor = 1
        if 'lr_mult' in optimizer_options:
            pattern, mult_factor = optimizer_options.pop('lr_mult')
            for name, param in decay.items():
                if re.match(pattern, name):
                    decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    decay_1x.append(param)
            for name, param in no_decay.items():
                if re.match(pattern, name):
                    no_decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    no_decay_1x.append(param)
            assert(len(decay_1x) + len(decay_mult) == len(decay))
            assert(len(no_decay_1x) + len(no_decay_mult) == len(no_decay))
        else:
            decay_1x, no_decay_1x = list(decay.values()), list(no_decay.values())
        wd = optimizer_options.pop('weight_decay', 0.)
        parameters = [
            {'params': no_decay_1x, 'weight_decay': 0.},
            {'params': decay_1x, 'weight_decay': wd},
            {'params': no_decay_mult, 'weight_decay': 0., 'lr': args.start_lr * mult_factor},
            {'params': decay_mult, 'weight_decay': wd, 'lr': args.start_lr * mult_factor},
        ]
        _logger.info('Parameters excluded from weight decay:\n - %s', '\n - '.join(names_no_decay))
        if len(names_lr_mult):
            _logger.info('Parameters with lr multiplied by %s:\n - %s', mult_factor, '\n - '.join(names_lr_mult))
    else:
        parameters = model.parameters()

    if args.optimizer == 'ranger':
        from weaver.utils.nn.optimizer.ranger import Ranger
        opt = Ranger(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'adamW':
        opt = torch.optim.AdamW(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'radam':
        opt = torch.optim.RAdam(parameters, lr=args.start_lr, **optimizer_options)

    # load previous training and resume if `--load-epoch` is set
    if args.load_epoch is not None:
        _logger.info('Resume training from epoch %d' % args.load_epoch)
        model_state = torch.load(args.model_prefix + '_epoch-%d_state.pt' % args.load_epoch, map_location=device)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        opt_state_file = args.model_prefix + '_epoch-%d_optimizer.pt' % args.load_epoch
        if os.path.exists(opt_state_file):
            opt_state = torch.load(opt_state_file, map_location=device)
            opt.load_state_dict(opt_state)
            if args.force_lr:
                for g in opt.param_groups:
                    g['lr'] = args.start_lr
        else:
            _logger.warning('Optimizer state file %s NOT found!' % opt_state_file)

    scheduler = None
    if args.lr_finder is None:
        if args.lr_scheduler == 'steps':
            lr_step = round(args.num_epochs / 3)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=[lr_step, 2 * lr_step], gamma=0.1,
                last_epoch=-1 if args.load_epoch is None else args.load_epoch)
        elif args.lr_scheduler == 'flat+decay':
            num_decay_epochs = max(1, int(args.num_epochs * 0.3))
            milestones = list(range(args.num_epochs - num_decay_epochs, args.num_epochs))
            gamma = 0.01 ** (1. / num_decay_epochs)
            if len(names_lr_mult):
                def get_lr(epoch): return gamma ** max(0, epoch - milestones[0] + 1)  # noqa
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    opt, (lambda _: 1, lambda _: 1, get_lr, get_lr),
                    last_epoch=-1 if args.load_epoch is None else args.load_epoch, verbose=True)
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    opt, milestones=milestones, gamma=gamma,
                    last_epoch=-1 if args.load_epoch is None else args.load_epoch)
        elif args.lr_scheduler == 'flat+linear' or args.lr_scheduler == 'flat+cos':
            total_steps = args.num_epochs * args.steps_per_epoch
            warmup_steps = args.warmup_steps
            flat_steps = total_steps * 0.7 - 1
            min_factor = 0.001

            def lr_fn(step_num):
                if step_num > total_steps:
                    raise ValueError(
                        "Tried to step {} times. The specified number of total steps is {}".format(
                            step_num + 1, total_steps))
                if step_num < warmup_steps:
                    return 1. * step_num / warmup_steps
                if step_num <= flat_steps:
                    return 1.0
                pct = (step_num - flat_steps) / (total_steps - flat_steps)
                if args.lr_scheduler == 'flat+linear':
                    return max(min_factor, 1 - pct)
                else:
                    return max(min_factor, 0.5 * (math.cos(math.pi * pct) + 1))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_fn, last_epoch=-1 if args.load_epoch is None else args.load_epoch * args.steps_per_epoch)
            scheduler._update_per_step = True  # mark it to update the lr every step, instead of every epoch
        elif args.lr_scheduler == 'one-cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.start_lr, epochs=args.num_epochs, steps_per_epoch=args.steps_per_epoch, pct_start=0.3,
                anneal_strategy='cos', div_factor=25.0, last_epoch=-1 if args.load_epoch is None else args.load_epoch)
            scheduler._update_per_step = True  # mark it to update the lr every step, instead of every epoch
    return opt, scheduler


def model_setup(args, data_config, dev):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config, name='_network_module')
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    _logger.info('Network options: %s' % str(network_options))
    if args.export_onnx:
        network_options['for_inference'] = True
    if args.use_amp:
        network_options['use_amp'] = True
    model, model_info = network_module.get_model(data_config, **network_options)
    if args.load_model_weights:
        model_state = torch.load(args.load_model_weights, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _logger.info('Model initialized with weights from %s\n ... Missing: %s\n ... Unexpected: %s' %
                     (args.load_model_weights, missing_keys, unexpected_keys))
    # _logger.info(model)
    flops(model, model_info)
    # loss function
    try:
        loss_func = network_module.get_loss(data_config, **network_options)
        aux_loss_func_clas = network_module.get_aux_loss_clas(data_config, dev, **network_options)
        aux_loss_func_regr = network_module.get_aux_loss_regr(data_config, **network_options)
        aux_loss_func_bin = network_module.get_aux_loss_bin(data_config, **network_options)
        _logger.info('Using loss function %s with options %s' % (loss_func, network_options))
    except AttributeError:
        loss_func = torch.nn.CrossEntropyLoss()
        aux_loss_func_clas = torch.nn.CrossEntropyLoss()
        aux_loss_func_regr = torch.nn.MSELoss()
        aux_loss_func_bin = torch.nn.BCEWithLogitsLoss()
        _logger.warning('Loss function not defined in %s. Will use `torch.nn.CrossEntropyLoss()`,`torch.nn.MSELoss()` and `torch.nn.BCEWithLogitsLoss()` by default.',
                        args.network_config)
    return model, model_info, loss_func, aux_loss_func_clas, aux_loss_func_regr, aux_loss_func_bin


def iotest(args, data_loader):
    """
    Io test
    :param args:
    :param data_loader:
    :return:
    """
    from tqdm.auto import tqdm
    from collections import defaultdict
    from weaver.utils.data.tools import _concat
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


def save_root(args, output_path, data_config, scores, labels, observers):
    """
    Saves as .root
    :param data_config:
    :param scores:
    :param labels
    :param observers
    :return:
    """
    from weaver.utils.data.fileio import _write_root
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
    _write_root(output_path, output)


def save_parquet(args, output_path, scores, labels, observers):
    """
    Saves as parquet file
    :param scores:
    :param labels:
    :param observers:
    :return:
    """
    import awkward as ak
    output = {'scores': scores}
    output.update(labels)
    output.update(observers)
    ak.to_parquet(ak.Array(output), output_path, compression='LZ4', compression_level=4)


def copy_log(args, epoch, type_log = ""):
    dirname=os.path.dirname(args.model_prefix)
    performance_dir=os.path.join(dirname, f'performance_{dirname.split("/")[-1].strip()}')
    log_name=os.path.join(performance_dir,
        f'{dirname.split("/")[-1].strip()}{epoch:02d}{type_log}.log')
    old_log_file=open(args.log).read()
    with open(log_name, 'w') as f:
        new_log = old_log_file[:old_log_file.index('Epoch #')]
        new_log += old_log_file[old_log_file.index(f'Epoch #{epoch} {type_log}'):]
        f.write(new_log)


    _logger.info('log file copied to: \n%s' % log_name)
    _logger.info('Performance data are stored in directory: \n%s/' % performance_dir)

def best_epoch_handler(args, best_epoch, best_valid_metric, valid_metric,
                       best_valid_comb_loss, valid_comb_loss,
                       best_valid_loss, valid_loss,
                       best_valid_aux_metric_pf, valid_aux_metric_pf,
                       best_valid_aux_dist, valid_aux_dist,
                       best_valid_aux_metric_pair, valid_aux_metric_pair,
                       best_valid_aux_loss, valid_aux_loss,
                       local_rank, epoch, roc_prefix, eval_type = 'validation', test = False,
                       best=False):

    is_best_epoch = (
        valid_metric < best_valid_metric) if args.regression_mode else(
        valid_metric > best_valid_metric)
    if (is_best_epoch and not test) or best:
        best_epoch=epoch
        best_valid_metric = valid_metric
        best_valid_comb_loss = valid_comb_loss
        best_valid_loss = valid_loss
        best_valid_aux_metric_pf= valid_aux_metric_pf
        best_valid_aux_dist= valid_aux_dist
        best_valid_aux_metric_pair= valid_aux_metric_pair
        best_valid_aux_loss = valid_aux_loss
        if not test:
            if args.model_prefix and (args.backend is None or local_rank == 0):
                shutil.copy2((args.model_prefix).split('_epoch')[0] + '_epoch-%d_state.pt' %
                                epoch, args.model_prefix + '_best_epoch_state.pt')
                # torch.save(model, args.model_prefix + '_best_epoch_full.pt')

    if test and best:
        #save labels for roc curve of best epoch
        for label_type in ["primary", "aux"]:
            save_labels_best_epoch(f'{roc_prefix}{label_type}_labels_epoch_{epoch:02d}.npz')

    _logger.info('Epoch #%d: Info saved in log file:\n%s' % (epoch, args.log))
    _logger.info('Best epoch: #%d' % (best_epoch))
    _logger.info('Epoch #%d: Current %s metric: %.5f (best: %.5f)  //  Current %s combined loss: %.5f (in best epoch: %.5f)  //  Current %s loss: %.5f (in best epoch: %.5f)' %
                    (epoch, eval_type, valid_metric, best_valid_metric, eval_type, valid_comb_loss,
                    best_valid_comb_loss, eval_type, valid_loss, best_valid_loss), color='bold')
    _logger.info('Epoch #%d: Current %s aux metric PF: %.5f (in best epoch: %.5f)  //  Current %s aux distance: %.5f (in best epoch: %.5f)  //  Current %s aux metric pair: %.5f (in best epoch: %.5f)  //  Current %s aux loss: %.5f (in best epoch: %.5f)' %
                    (epoch, eval_type, valid_aux_metric_pf, best_valid_aux_metric_pf,
                    eval_type, valid_aux_dist, best_valid_aux_dist, eval_type, valid_aux_metric_pair,
                    best_valid_aux_metric_pair, eval_type, valid_aux_loss, best_valid_aux_loss), color='bold')

    return best_epoch, best_valid_metric, best_valid_loss, best_valid_comb_loss, \
        best_valid_aux_metric_pf, best_valid_aux_dist, best_valid_aux_loss,\
        best_valid_aux_metric_pair

def get_best_metrics(args, last_epoch, best_epoch, best_valid_metric, best_valid_loss, best_valid_comb_loss, \
        best_valid_aux_metric_pf, best_valid_aux_dist, best_valid_aux_loss,\
        best_valid_aux_metric_pair, test):
    dirname = os.path.dirname(args.model_prefix)
    suffix=dirname.split('/')[-1].strip()
    performance_dir=os.path.join(dirname, f'performance_{suffix}', '')
    for file in os.listdir(performance_dir):
        if f'{last_epoch:02d}val.log' in file:
            with open(os.path.join(performance_dir,file)) as f:
                f = f.readlines()
            for line in f:
                try:
                    if 'Best epoch' in line:
                        best_epoch=int(line.split(': #',1)[1].split('\n')[0])
                        if test: break
                    elif 'validation metric' in line :
                        best_valid_metric=float(line.split('(best: ',1)[1].split(')')[0])
                        best_valid_comb_loss=float(line.split('(in best epoch: ')[1].split(')')[0])
                        best_valid_loss=float(line.split('(in best epoch: ')[2].split(')')[0])
                    elif 'validation aux metric' in line :
                        best_valid_aux_metric_pf=float(line.split('(in best epoch: ')[1].split(')')[0])
                        best_valid_aux_dist=float(line.split('(in best epoch: ')[2].split(')')[0])
                        best_valid_aux_metric_pair=float(line.split('(in best epoch: ')[3].split(')')[0])
                        best_valid_aux_loss=float(line.split('(in best epoch: ')[4].split(')')[0])
                except IndexError:
                    pass
    return best_epoch, best_valid_metric, best_valid_loss, best_valid_comb_loss, \
        best_valid_aux_metric_pf, best_valid_aux_dist, best_valid_aux_loss,\
        best_valid_aux_metric_pair

def _main(args):
    _logger.info('args:\n - %s', '\n - '.join(str(it) for it in args.__dict__.items()))

    # export to ONNX
    if args.export_onnx:
        onnx(args)
        return

    if args.file_fraction < 1:
        _logger.warning('Use of `file-fraction` is not recommended in general -- prefer using `data-fraction` instead.')

    # classification/regression mode
    if args.regression_mode:
        _logger.info('Running in regression mode')
        from weaver.utils.nn.tools import train_regression as train
        from weaver.utils.nn.tools import evaluate_regression as evaluate
    else:
        _logger.info('Running in classification mode')
        from weaver.utils.nn.tools import train_classification as train
        from weaver.utils.nn.tools import evaluate_classification as evaluate

    # training/testing mode
    training_mode = not args.predict and  args.train

    if args.train and args.test and not args.val:
        raise RuntimeError("Please test only on already trained and validated models")
    local_rank = None
    # device
    if args.gpus:
        # distributed training
        if args.backend is not None:
            local_rank = args.local_rank
            torch.cuda.set_device(local_rank)
            gpus = [local_rank]
            dev = torch.device(local_rank)
            torch.distributed.init_process_group(backend=args.backend)
            _logger.info(f'Using distributed PyTorch with {args.backend} backend')
        else:
            gpus = [int(i) for i in args.gpus.split(',')]
            dev = torch.device(gpus[0])
    else:
        gpus = None
        dev = torch.device('cpu')

    # load data
    if args.train or args.val:
        train_loader, val_loader, data_config, train_input_names, train_label_names, train_aux_label_names = train_load(args)
    elif args.predict or args.test:
        test_loaders, data_config = test_load(args)

    if args.io_test:
        data_loader = train_loader if training_mode else list(test_loaders.values())[0]()
        iotest(args, data_loader)
        return

    model, model_info, loss_func, aux_loss_func_clas, aux_loss_func_regr, aux_loss_func_bin = model_setup(args, data_config, dev)

    # TODO: load checkpoint
    # if args.backend is not None:
    #     load_checkpoint()

    if args.print:
        return

    if args.profile:
        profile(args, model, model_info, device=dev)
        return

    if args.tensorboard:
        from weaver.utils.nn.tools import TensorboardHelper
        tb = TensorboardHelper(tb_comment=args.tensorboard, tb_custom_fn=args.tensorboard_custom_fn)
    else:
        tb = None

    # note: we should always save/load the state_dict of the original model, not the one wrapped by nn.DataParallel
    # so we do not convert it to nn.DataParallel now
    orig_model = model

    best_epoch, best_valid_metric, best_valid_loss, best_valid_comb_loss, \
    best_valid_aux_metric_pf, best_valid_aux_dist, best_valid_aux_loss,\
    best_valid_aux_metric_pair = -1, np.inf if args.regression_mode else -1, 0,0,0,0,0,0

    last_epoch = -1
    test = False
    if args.load_epoch is not None and args.train:
        last_epoch = args.load_epoch
        test=False
    elif args.val and not args.train:
        try:
            last_epoch = int(args.val_epochs[:2]) - 1
        except:
            last_epoch = int(args.val_epochs[0]) - 1
        test=False
    elif args.test and not (args.train or args.val):
        dirname = os.path.dirname(args.model_prefix)
        suffix=dirname.split('/')[-1].strip()
        performance_dir=os.path.join(dirname, f'performance_{suffix}', '')
        #print(os.listdir(performance_dir))
        for file in sorted(os.listdir(performance_dir)):
            if 'val.log'in file:
                last_epoch = int(file.split('val')[0][-2:])
        test=True
    if last_epoch != -1:
        best_epoch, best_valid_metric, best_valid_loss, best_valid_comb_loss, \
        best_valid_aux_metric_pf, best_valid_aux_dist, best_valid_aux_loss,\
        best_valid_aux_metric_pair = get_best_metrics(args, last_epoch, \
            best_epoch, best_valid_metric, best_valid_loss, best_valid_comb_loss, \
            best_valid_aux_metric_pf, best_valid_aux_dist, best_valid_aux_loss,\
            best_valid_aux_metric_pair, test)



    if training_mode:
        model = orig_model.to(dev)

        # DistributedDataParallel
        if args.backend is not None:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpus, output_device=local_rank)

        # optimizer & learning rate
        opt, scheduler = optim(args, model, dev)

        # DataParallel
        if args.backend is None:
            if gpus is not None and len(gpus) > 1:
                # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
                model = torch.nn.DataParallel(model, device_ids=gpus)
            # model = model.to(dev)

        # lr finder: keep it after all other setups
        if args.lr_finder is not None:
            start_lr, end_lr, num_iter = args.lr_finder.replace(' ', '').split(',')
            from weaver.utils.lr_finder import LRFinder
            lr_finder = LRFinder(model, opt, loss_func, device=dev, input_names=train_input_names,
                                 label_names=train_label_names)
            # HERE lr_finder is a complementary feature
            # maybe put lr_finder where instead of labels put aux_labels and change loss function
            lr_finder.range_test(train_loader, start_lr=float(start_lr), end_lr=float(end_lr), num_iter=int(num_iter))
            lr_finder.plot(output='lr_finder.png')  # to inspect the loss-learning rate graph
            return

        # training loop
        grad_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        for epoch in range(args.num_epochs):
            if args.load_epoch is not None:
                if epoch <= args.load_epoch:
                    continue

            if epoch >= args.no_aux_epoch and not args.aux_saturation:
                aux_weight = 0
            elif epoch >= args.no_aux_epoch and args.aux_saturation:
                aux_weight= 1/(args.no_aux_epoch)
            else:
                aux_weight = 1/(epoch+1)

            _logger.info('-' * 50)
            if not args.val:
                _logger.info('Epoch #%d training only' % epoch)
            else:
                _logger.info('Epoch #%d training' % epoch)

            if args.model_prefix and (args.backend is None or local_rank == 0):
                dirname = os.path.dirname(args.model_prefix)
                suffix=dirname.split('/')[-1].strip()
                performance_dir=os.path.join(dirname, f'performance_{suffix}', '')
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                if performance_dir and not os.path.exists(performance_dir):
                    os.makedirs(performance_dir)
                roc_prefix=os.path.join(performance_dir,suffix)

            train(model, loss_func, aux_loss_func_clas, aux_loss_func_regr,
                  aux_loss_func_bin, opt, scheduler, train_loader, dev, epoch, aux_weight,
                  steps_per_epoch=args.steps_per_epoch, grad_scaler=grad_scaler,
                  tb_helper=tb, epoch_division = args.epoch_division)


            if args.model_prefix and (args.backend is None or local_rank == 0):
                state_dict = model.module.state_dict() if isinstance(
                    model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict()
                torch.save(state_dict, args.model_prefix + '_epoch-%d_state.pt' % epoch)
                torch.save(opt.state_dict(), args.model_prefix + '_epoch-%d_optimizer.pt' % epoch)
            # if args.backend is not None and local_rank == 0:
            # TODO: save checkpoint
            #     save_checkpoint()

            copy_log(args, epoch, 'train')

            if args.val:
                _logger.info('Epoch #%d validating' % epoch)
                valid_metric, valid_comb_loss, valid_loss, valid_aux_metric_pf,\
                valid_aux_dist, valid_aux_metric_pair, valid_aux_loss = \
                        evaluate(model, val_loader, dev, epoch, aux_weight, loss_func=loss_func,
                            aux_loss_func_clas=aux_loss_func_clas, aux_loss_func_regr=aux_loss_func_regr,
                            aux_loss_func_bin=aux_loss_func_bin,
                            steps_per_epoch=args.steps_per_epoch_val, tb_helper=tb, roc_prefix=roc_prefix, epoch_division=args.epoch_division)

                best_epoch, best_valid_metric, best_valid_loss, best_valid_comb_loss, \
                best_valid_aux_metric_pf, best_valid_aux_dist, best_valid_aux_loss,\
                best_valid_aux_metric_pair = best_epoch_handler(args, best_epoch,
                        best_valid_metric, valid_metric,
                        best_valid_comb_loss, valid_comb_loss,
                        best_valid_loss, valid_loss,
                        best_valid_aux_metric_pf, valid_aux_metric_pf,
                        best_valid_aux_dist, valid_aux_dist,
                        best_valid_aux_metric_pair, valid_aux_metric_pair,
                        best_valid_aux_loss, valid_aux_loss,
                        local_rank, epoch, roc_prefix)
                copy_log(args,epoch, 'val')

    if args.data_test:
        if args.backend is not None and local_rank != 0:
            return
        try:
            del train_loader
        except UnboundLocalError:
            pass
        if args.val and args.train: del val_loader


        if not args.model_prefix.endswith('.onnx'):
            if args.predict_gpus:
                gpus = [int(i) for i in args.predict_gpus.split(',')]
                dev = torch.device(gpus[0])
            else:
                gpus = None
                dev = torch.device('cpu')
            model = orig_model.to(dev)

            if ',' in args.val_epochs:
                val_epochs = [int(i) for i in args.val_epochs.split(',')]
            elif ':' in args.val_epochs:
                val_epochs_ext= [int(i) for i in args.val_epochs.split(':')]
                val_epochs = [i for i in range(val_epochs_ext[0], val_epochs_ext[1]+1)]
            elif args.val_epochs == '-1':
                val_epoch_len = len([filename for filename in os.listdir(os.path.dirname(args.model_prefix))\
                                    if '_state.pt' in filename and 'best' not in filename])
                val_epochs = [int(i) for i in range(val_epoch_len)]
            else:
                val_epochs = [int(args.val_epochs)]

        if args.model_prefix and (args.backend is None or local_rank == 0):
            dirname = os.path.dirname(args.model_prefix)
            suffix=dirname.split('/')[-1].strip()
            performance_dir=os.path.join(dirname, f'performance_{suffix}', '')
            if performance_dir and not os.path.exists(performance_dir):
                os.makedirs(performance_dir)
            roc_prefix=os.path.join(performance_dir,suffix)

            if args.val and not args.train:
                performance_files = os.listdir(performance_dir)

                for epoch in val_epochs:
                    model_path = f'{args.model_prefix}_epoch-{epoch}_state.pt'
                    _logger.info('Loading model %s for eval' % model_path)
                    model.load_state_dict(torch.load(model_path, map_location=dev))
                    if gpus is not None and len(gpus) > 1:
                        model = torch.nn.DataParallel(model, device_ids=gpus)
                    model = model.to(dev)
                    _logger.info('Epoch #%d validating only' % epoch)

                    if epoch >= args.no_aux_epoch and not args.aux_saturation:
                        aux_weight = 0
                    elif epoch >= args.no_aux_epoch and args.aux_saturation:
                        aux_weight= 1/(args.no_aux_epoch)
                    else:
                        aux_weight = 1/(epoch+1)

                    valid_metric, valid_comb_loss, valid_loss, valid_aux_metric_pf,\
                    valid_aux_dist, valid_aux_metric_pair, valid_aux_loss = \
                    evaluate(model, val_loader, dev, epoch, aux_weight,loss_func=loss_func,
                            aux_loss_func_clas=aux_loss_func_clas, aux_loss_func_regr=aux_loss_func_regr,
                            aux_loss_func_bin=aux_loss_func_bin,
                            steps_per_epoch=args.steps_per_epoch_val, tb_helper=tb,
                            roc_prefix=roc_prefix, epoch_division=args.epoch_division)

                    best_epoch, best_valid_metric, best_valid_loss, best_valid_comb_loss, \
                    best_valid_aux_metric_pf, best_valid_aux_dist, best_valid_aux_loss,\
                    best_valid_aux_metric_pair = best_epoch_handler(args, best_epoch,
                        best_valid_metric, valid_metric,
                        best_valid_comb_loss, valid_comb_loss,
                        best_valid_loss, valid_loss,
                        best_valid_aux_metric_pf, valid_aux_metric_pf,
                        best_valid_aux_dist, valid_aux_dist,
                        best_valid_aux_metric_pair, valid_aux_metric_pair,
                        best_valid_aux_loss, valid_aux_loss,
                        local_rank, epoch, roc_prefix)
                    copy_log(args, epoch, "val")

            elif not args.test and not training_mode:
                for name, get_test_loader in test_loaders.items():
                    test_loader = get_test_loader()
                    # run prediction
                    if args.model_prefix.endswith('.onnx'):
                        _logger.info('Loading model %s for eval' % args.model_prefix)
                        from weaver.utils.nn.tools import evaluate_onnx
                        test_metric, scores, labels, observers = evaluate_onnx(args.model_prefix, test_loader,epoch=-1, roc_prefix=roc_prefix)
                    else:
                            test_metric, test_loss, scores, labels, observers = evaluate(
                                model, test_loader, dev, epoch=-1, for_training=False, tb_helper=tb, roc_prefix=roc_prefix)
                    _logger.info('Test metric %.5f   //   Test loss %.5f' % (test_metric, test_loss), color='bold')
                    del test_loader

                    if args.predict_output:
                        if '/' not in args.predict_output:
                            predict_output = os.path.join(
                                os.path.dirname(args.model_prefix),
                                'predict_output', args.predict_output)
                        else:
                            predict_output = args.predict_output
                        os.makedirs(os.path.dirname(predict_output), exist_ok=True)
                        if name == '':
                            output_path = predict_output
                        else:
                            base, ext = os.path.splitext(predict_output)
                            output_path = base + '_' + name + ext
                        if output_path.endswith('.root'):
                            save_root(args, output_path, data_config, scores, labels, observers)
                        else:
                            save_parquet(args, output_path, scores, labels, observers)
                        _logger.info('Written output to %s' % output_path, color='bold')

            if args.test:
                try:
                    del val_loader
                except UnboundLocalError:
                    pass
                if args.val:
                    test_loaders, data_config = test_load(args)

                best_valid_metric, best_valid_loss, best_valid_comb_loss, \
                best_valid_aux_metric_pf, best_valid_aux_dist, best_valid_aux_loss,\
                best_valid_aux_metric_pair = np.inf if args.regression_mode else -1, 0,0,0,0,0,0

                if not args.test_epochs:
                    test_epochs = val_epochs
                elif ',' in args.test_epochs:
                    test_epochs = [int(i) for i in args.test_epochs.split(',')]
                elif ':' in args.test_epochs:
                    test_epochs_ext= [int(i) for i in args.test_epochs.split(':')]
                    test_epochs = [i for i in range(test_epochs_ext[0], test_epochs_ext[1]+1)]
                elif args.test_epochs == '-1':
                    test_epoch_len = len([filename for filename in os.listdir(os.path.dirname(args.model_prefix))\
                                        if '_state.pt' in filename and 'best' not in filename])
                    test_epochs = [int(i) for i in range(test_epoch_len)]
                elif args.test_epochs == 'best+last':
                    last_epoch = len([filename for filename in os.listdir(os.path.dirname(args.model_prefix))\
                                        if '_state.pt' in filename and 'best' not in filename])-1
                    test_epochs = [last_epoch]
                elif args.test_epochs == 'best':
                    test_epochs = [best_epoch]
                else:
                    test_epochs = [int(args.test_epochs)]
                if best_epoch not in test_epochs: test_epochs.append(best_epoch)

                performance_files = os.listdir(performance_dir)

                # Test on epoch
                for epoch in test_epochs:
                    model_path = f'{args.model_prefix}_epoch-{epoch}_state.pt'
                    _logger.info('Loading model %s for test' % model_path)
                    model.load_state_dict(torch.load(model_path, map_location=dev))
                    if gpus is not None and len(gpus) > 1:
                        model = torch.nn.DataParallel(model, device_ids=gpus)
                    model = model.to(dev)
                    _logger.info('Epoch #%d testing' % epoch)

                    if epoch >= args.no_aux_epoch and not args.aux_saturation:
                        aux_weight = 0
                    elif epoch >= args.no_aux_epoch and args.aux_saturation:
                        aux_weight= 1/(args.no_aux_epoch)
                    else:
                        aux_weight = 1/(epoch+1)

                    for name, get_test_loader in test_loaders.items():
                        test_loader = get_test_loader()
                        valid_metric, valid_comb_loss, valid_loss, valid_aux_metric_pf,\
                        valid_aux_dist, valid_aux_metric_pair, valid_aux_loss = \
                                evaluate(model, test_loader, dev, epoch, aux_weight,loss_func=loss_func,
                                    aux_loss_func_clas=aux_loss_func_clas, aux_loss_func_regr=aux_loss_func_regr,
                                    aux_loss_func_bin=aux_loss_func_bin,
                                    steps_per_epoch=args.steps_per_epoch_val, tb_helper=tb, roc_prefix=roc_prefix,
                                    eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix', 'save_labels'],
                                    eval_aux_metrics = ['aux_confusion_matrix_pf_clas', 'aux_confusion_matrix_pair_bin', 'aux_save_labels'],
                                    type_eval='test', epoch_division=args.epoch_division)

                        best_epoch, best_valid_metric, best_valid_loss, best_valid_comb_loss, \
                        best_valid_aux_metric_pf, best_valid_aux_dist, best_valid_aux_loss,\
                        best_valid_aux_metric_pair = best_epoch_handler(args, best_epoch,
                                best_valid_metric, valid_metric,
                                best_valid_comb_loss, valid_comb_loss,
                                best_valid_loss, valid_loss,
                                best_valid_aux_metric_pf, valid_aux_metric_pf,
                                best_valid_aux_dist, valid_aux_dist,
                                best_valid_aux_metric_pair, valid_aux_metric_pair,
                                best_valid_aux_loss, valid_aux_loss,
                                local_rank, epoch, roc_prefix, eval_type='test',
                                test = True, best=(epoch == best_epoch))
                        copy_log(args, epoch, "test")
                        del test_loader


def main():
    args = parser.parse_args()

    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size
        else:
            raise RuntimeError('Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!')

    if args.samples_per_epoch_val is not None:
        if args.steps_per_epoch_val is None:
            args.steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size
        else:
            raise RuntimeError('Please use either `--steps-per-epoch-val` or `--samples-per-epoch-val`, but not both!')

    if args.steps_per_epoch_val is None and args.steps_per_epoch is not None:
        args.steps_per_epoch_val = round(args.steps_per_epoch * (1 - args.train_val_split) / args.train_val_split)
    if args.steps_per_epoch_val is not None and args.steps_per_epoch_val < 0:
        args.steps_per_epoch_val = None



    if '{auto}' in args.model_prefix or '{auto}' in args.log:
        import hashlib
        import time
        model_name = time.strftime('%Y%m%d-%H%M%S') + "_" + os.path.basename(args.data_config).replace('.yaml', '')
        if len(args.network_option):
            model_name = model_name + "_" + hashlib.md5(str(args.network_option).encode('utf-8')).hexdigest()
        model_name += '_{optim}_lr{lr}_batch{batch}'.format(lr=args.start_lr,
                                                            optim=args.optimizer, batch=args.batch_size)
        args._auto_model_name = model_name
        args.model_prefix = args.model_prefix.replace('{auto}', model_name)
        args.log = args.log.replace('{auto}', model_name)
        print('Using auto-generated model prefix %s' % args.model_prefix)

    if (args.val and not args.train) or (args.test and not args.train):
        if ',' in  args.gpus:
            args.gpus = '0'
        args.extra_selection = "(np.abs(jet_eta)<1.4) & (jet_pt>30) & (jet_pt<200)"
        args.extra_test_selection = "(np.abs(jet_eta)<1.4) & (jet_pt>30) & (jet_pt<200)"
        print('Using extra selection: %s' % args.extra_selection)
        print('Using extra test selection: %s' % args.extra_test_selection)

    if args.predict_gpus is None:
        args.predict_gpus = args.gpus

    args.local_rank = None if args.backend is None else int(os.environ.get("LOCAL_RANK", "0"))

    stdout = sys.stdout
    if args.local_rank is not None:
        args.log += '.%03d' % args.local_rank
        if args.local_rank != 0:
            stdout = None
    _configLogger('weaver', stdout=stdout, filename=args.log)

    if args.cross_validation:
        model_dir, model_fn = os.path.split(args.model_prefix)
        var_name, kfold = args.cross_validation.split('%')
        kfold = int(kfold)
        for i in range(kfold):
            _logger.info(f'\n=== Running cross validation, fold {i} of {kfold} ===')
            args.model_prefix = os.path.join(f'{model_dir}_fold{i}', model_fn)
            args.extra_selection = f'{var_name}%{kfold}!={i}'
            args.extra_test_selection = f'{var_name}%{kfold}=={i}'
            _main(args)
    else:
        _main(args)

    _logger.info('Finished!\n\n')


if __name__ == '__main__':
    main()
