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
from utils.logger import _logger
from utils.dataset import SimpleIterDataset
from utils.nn.tools import train, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--data-config', type=str, default='data/ak15_points_pf_sv_v0.yaml',
                        help='data config YAML file')
    parser.add_argument('-i', '--data-train', nargs='*', default=[],
                        help='training files')
    parser.add_argument('-t', '--data-test', nargs='*', default=[],
                        help='testing files')
    parser.add_argument('--data-fraction', type=float, default=1,
                        help='fraction of events to load from each file; for training, the events are randomly selected for each epoch')
    parser.add_argument('--data-dilation', type=int, default=1,
                        help='reduce number of file by a factor of `d` for training. NOT recommended in general - use `--data-fraction` instead.')
    parser.add_argument('--files-per-fetch', type=int, default=20,
                        help='number of files to load each time; shuffling is done within these events, so choose a number large enough to get events from all classes')
    parser.add_argument('--train-val-split', type=float, default=0.8,
                        help='training/validation split fraction')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='quickly test the setup by running over only a small number of events')
    parser.add_argument('--lr-finder', type=str, default=None,
                        help='run learning rate finder instead of the actual training; format: ``start_lr, end_lr, num_iters``')
    parser.add_argument('-n', '--network-config', type=str, default='networks/particle_net_pfcand_sv.py',
                        help='network architecture configuration file; the path must be relative to the current dir')
    parser.add_argument('--network-option', nargs=2, action='append', default=[],
                        help='options to pass to the model class constructor, e.g., `--network-option use_counts False`')
    parser.add_argument('-m', '--model-prefix', type=str, default='test_output/model_name',
                        help='path to save or load the model; for training, this will be used as a prefix; for testing, this should be the full path including extension')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='ranger', choices=['adam', 'ranger'],  # TODO: add more
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
                        help='device for the training/testing; to use CPU, set to empty string (''); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='number of threads to load the dataset; memory consuption and disk access load increases (~linearly) with this numbers')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='run prediction instead of training')
    parser.add_argument('--predict-output', type=str,
                        help='path to save the prediction output, support `.root` and `.awkd` format')
    parser.add_argument('--export-onnx', type=str, default=None,
                        help='export the PyTorch model to ONNX model and save it at the given path (path must ends w/ .onnx); '
                        'needs to set `--data-config`, `--network-config`, and `--model-prefix` (requires the full model path)')

    args = parser.parse_args()
    _logger.info(args)

    if args.use_amp:
        raise NotImplementedError
#         from apex import amp

    if args.data_dilation > 1:
        _logger.warning('Use of `data-dilation` is not recomended in general -- consider using `data-fraction` instead.')

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
        filelist = sorted(sum([glob.glob(f) for f in args.data_train], []))
        # np.random.seed(1)
        np.random.shuffle(filelist)
        if args.demo:
            filelist = filelist[:20]
            _logger.info(filelist)
            args.data_fraction = 0.1
            args.files_per_fetch = 5
        train_data = SimpleIterDataset(filelist, args.data_config, for_training=True, load_range_and_fraction=((0, args.train_val_split), args.data_fraction),
                                       dilation=args.data_dilation, files_per_fetch=args.files_per_fetch)
        val_data = SimpleIterDataset(filelist, args.data_config, for_training=True, load_range_and_fraction=((args.train_val_split, 1), args.data_fraction),
                                     dilation=args.data_dilation, files_per_fetch=args.files_per_fetch)
        train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_data, num_workers=args.num_workers, batch_size=args.batch_size, drop_last=True, pin_memory=True)
        data_config = train_data.config
    else:
        filelist = sorted(sum([glob.glob(f) for f in args.data_test], []))
        test_data = SimpleIterDataset(filelist, args.data_config, for_training=False, files_per_fetch=1)
        test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size, drop_last=False, pin_memory=True)
        data_config = test_data.config

    # model
    network_module = import_module(args.network_config.replace('.py', '').replace('/', '.'))
    network_options = {k:ast.literal_eval(v) for k, v in args.network_option}
    if args.export_onnx:
        network_options['for_inference'] = True
    model, model_info = network_module.get_model(data_config, **network_options)
    _logger.info(model)

    # export to ONNX
    if args.export_onnx:
        assert(args.export_onnx.endswith('.onnx'))
        model_path = args.model_prefix
        _logger.info('Exporting model %s to ONNX' % model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.cpu()
        model.eval()

        os.makedirs(os.path.dirname(args.export_onnx), exist_ok=True)
        inputs = tuple(torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])
        torch.onnx.export(model, inputs, args.export_onnx,
                          input_names=model_info['input_names'],
                          output_names=model_info['output_names'],
                          dynamic_axes=model_info.get('dynamic_axes', None),
                          opset_version=11)
        _logger.info('ONNX model saved to %s', args.export_onnx)
        return

    # note: we should always save/load the state_dict of the original model, not the one wrapped by nn.DataParallel
    # so we do not convert it to nn.DataParallel now
    model = model.to(dev)

    # loss function
    try:
        loss_func = network_module.get_loss(data_config, **network_options)
        _logger.info(loss_func)
    except AttributeError:
        loss_func = torch.nn.CrossEntropyLoss()
        _logger.warning('Loss function not defined in %s. Will use `torch.nn.CrossEntropyLoss()` by default.', args.network_config)

    if training_mode:
        # optimizer & learning rate
        if args.optimizer == 'adam':
            opt = torch.optim.Adam(model.parameters(), lr=args.start_lr)
            if args.lr_finder is None:
                lr_steps = [int(x) for x in args.lr_steps.split(',')]
                scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_steps, gamma=0.1)
        else:
            from utils.nn.optimizer.ranger import Ranger
            opt = Ranger(model.parameters(), lr=args.start_lr)
            if args.lr_finder is None:
                lr_decay_epochs = max(1, int(args.num_epochs * 0.3))
                lr_decay_rate = 0.01 ** (1. / lr_decay_epochs)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(range(args.num_epochs - lr_decay_epochs, args.num_epochs)), gamma=lr_decay_rate)

        # TODO: mixed precision training
        if args.use_amp:
#             model, opt = amp.initialize(
#                model, opt, opt_level="O2",
#                keep_batchnorm_fp32=True, loss_scale="dynamic"
#             )
            model, opt = amp.initialize(
               model, opt, opt_level="O1",
               keep_batchnorm_fp32=None, loss_scale="dynamic"
            )

        # load previous training and resume if `--load-epoch` is set
        if args.load_epoch is not None:
            _logger.info('Resume training from epoch %d' % args.load_epoch)
            model_state = torch.load(args.model_prefix + '_epoch-%d_state.pt' % args.load_epoch, map_location=dev)
            model.load_state_dict(model_state)
            opt_state = torch.load(args.model_prefix + '_epoch-%d_optimizer.pt' % args.load_epoch, map_location=dev)
            opt.load_state_dict(opt_state)

        # mutli-gpu
        if gpus is not None and len(gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpus)  # model becomes `torch.nn.DataParallel` w/ model.module being the orignal `torch.nn.Module`
        model = model.to(dev)

        # lr finder: keep it after all other setups
        if args.lr_finder is not None:
            start_lr, end_lr, num_iter = args.lr_finder.replace(' ', '').split(',')
            from utils.lr_finder import LRFinder
            lr_finder = LRFinder(model, opt, loss_func, device=dev, input_names=train_data.config.input_names, label_names=train_data.config.label_names)
            lr_finder.range_test(train_loader, start_lr=float(start_lr), end_lr=float(end_lr), num_iter=int(num_iter))
            lr_finder.plot(output='lr_finder.png')  # to inspect the loss-learning rate graph
            return

        # training loop
        best_valid_acc = 0
        for epoch in range(args.num_epochs):
            if args.load_epoch is not None:
                if epoch <= args.load_epoch:
                    continue
            print('-' * 50)
            _logger.info('Epoch #%d training' % epoch)
            train(model, loss_func, opt, scheduler, train_loader, dev)
            if args.model_prefix:
                dirname = os.path.dirname(args.model_prefix)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_dict, args.model_prefix + '_epoch-%d_state.pt' % epoch)
                torch.save(opt.state_dict(), args.model_prefix + '_epoch-%d_optimizer.pt' % epoch)

            _logger.info('Epoch #%d validating' % epoch)
            valid_acc = evaluate(model, val_loader, dev, loss_func=loss_func)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if args.model_prefix:
                    shutil.copy2(args.model_prefix + '_epoch-%d_state.pt' % epoch, args.model_prefix + '_best_acc_state.pt')
                    torch.save(model, args.model_prefix + '_best_acc_full.pt')
            _logger.info('Epoch #%d: Current validation acc: %.5f (best: %.5f)' % (epoch, valid_acc, best_valid_acc))
    else:
        # run prediction
        if args.model_prefix.endswith('.onnx'):
            _logger.info('Loading model %s for eval' % args.model_prefix)
            from utils.nn.tools import evaluate_onnx
            test_acc, scores, labels, observers = evaluate_onnx(args.model_prefix, test_loader)
        else:
            model_path = args.model_prefix if args.model_prefix.endswith('.pt') else args.model_prefix + '_best_acc_state.pt'
            _logger.info('Loading model %s for eval' % model_path)
            model.load_state_dict(torch.load(model_path, map_location=dev))
            if gpus is not None and len(gpus) > 1:
                model = torch.nn.DataParallel(model, device_ids=gpus)
            model = model.to(dev)
            test_acc, scores, labels, observers = evaluate(model, test_loader, dev, for_training=False)
        _logger.info('Test acc %.5f' % test_acc)

        if args.predict_output:
            os.makedirs(os.path.dirname(args.predict_output), exist_ok=True)
            if args.predict_output.endswith('.root'):
                from utils.data.fileio import _write_root
                output = {}
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
            else:
                import awkward
                output = {'scores':scores}
                output.update(labels)
                output.update(observers)
                awkward.save(args.predict_output, output, mode='w')

            _logger.info('Written output to %s' % args.predict_output)


if __name__ == '__main__':
    main()
