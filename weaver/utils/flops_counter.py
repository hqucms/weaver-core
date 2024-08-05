'''
Copyright (C) 2019 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys
import traceback
from functools import partial
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .logger import _logger

'''Modified from: https://github.com/sovrasov/flops-counter.pytorch'''


def get_model_complexity_info(model, inputs,
                              print_per_layer_stat=True,
                              as_strings=True,
                              ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={},
                              output_precision=3,
                              flops_units: Optional[str] = None,
                              param_units: Optional[str] = 'M',
                              extra_config: Dict = {}) -> Tuple[Union[int, None],
                                                                Union[int, None]]:
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=ost, verbose=verbose,
                                  ignore_list=ignore_modules)

    enable_func_ops_patching = extra_config.get('count_functional', True)
    torch_functional_flops = []
    torch_tensor_ops_flops = []
    if enable_func_ops_patching:
        patch_functional(torch_functional_flops)
        patch_tensor_ops(torch_tensor_ops_flops)

    def reset_environment():
        flops_model.stop_flops_count()
        if enable_func_ops_patching:
            unpatch_functional()
            unpatch_tensor_ops()
        global CUSTOM_MODULES_MAPPING
        CUSTOM_MODULES_MAPPING = {}

    try:
        _ = flops_model(*inputs)
        flops_count, params_count = flops_model.compute_average_flops_cost()
        flops_count += sum(torch_functional_flops)
        flops_count += sum(torch_tensor_ops_flops)

    except Exception as e:
        _logger.info("Flops estimation was not finished successfully because of"
              f" the following exception:\n{type(e)} : {e}")
        traceback.print_exc()
        reset_environment()

        return None, None

    if print_per_layer_stat:
        print_model_with_flops(
            flops_model,
            flops_count,
            params_count,
            ost=ost,
            flops_units=flops_units,
            param_units=param_units,
            precision=output_precision
        )
    reset_environment()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops: int, units: Optional[str] = None, precision: int = 2) -> str:
    """
    Converts integer MACs representation to a readable string.

    :param flops: Input MACs.
    :param units: Units for string representation of MACs (GMac, MMac or KMac).
    :param precision: Floating point precision for representing MACs in
        given units.
    """
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num: int, units: Optional[str] = None,
                     precision: int = 2) -> str:
    """
    Converts integer params representation to a readable string.

    :param flops: Input number of parameters.
    :param units: Units for string representation of params (M, K or B).
    :param precision: Floating point precision for representing params in
        given units.
    """
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, precision)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, precision)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        elif units == 'B':
            return str(round(params_num / 10.**9, precision)) + ' ' + units
        else:
            return str(params_num)


def accumulate_flops(self):
    if is_supported_instance(self):
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops()
        return sum


def print_model_with_flops(model, total_flops, total_params,
                           flops_units: Optional[str] = 'GMac',
                           param_units: Optional[str] = 'M',
                           precision=3, ost=sys.stdout):
    if total_flops < 1:
        total_flops = 1
    if total_params < 1:
        total_params = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        if accumulated_params_num > total_params:
            _logger.info('Warning: parameters of some of the modules were counted twice because'
                  ' of multiple links to the same modules.'
                  ' Extended per layer parameters num statistic could be unreliable.')

        prefix = self.original_extra_repr() + ', |' if self.original_extra_repr() else '|'
        return prefix + ', '.join([
            params_to_string(accumulated_params_num, units=param_units, precision=precision),
            '{:.3%} Params'.format(accumulated_params_num / total_params),
            flops_to_string(accumulated_flops_cost, units=flops_units, precision=precision),
            '{:.3%} MACs'.format(accumulated_flops_cost / total_flops)]) + '|'

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    _logger.info(repr(model), color='lightgray')
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
        net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    for m in self.modules():
        m.accumulate_flops = accumulate_flops.__get__(m)

    flops_sum = self.accumulate_flops()

    for m in self.modules():
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    params_sum = get_model_parameters_number(self)
    return flops_sum / self.__batch_counter__, params_sum


def start_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_flops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                    CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                _logger.info('Warning: module ' + type(module).__name__ +
                             ' is treated as a zero-op.', color='lightgray')
            seen_types.add(type(module))

    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)
    self.apply(remove_flops_counter_variables)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        _logger.info('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__params__'):
            _logger.info('Warning: variables __flops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptflops can affect your code!')
            module.__ptflops_backup_flops__ = module.__flops__
            module.__ptflops_backup_params__ = module.__params__
        module.__flops__ = 0
        module.__params__ = get_model_parameters_number(module)


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


def remove_flops_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__'):
            del module.__flops__
            if hasattr(module, '__ptflops_backup_flops__'):
                module.__flops__ = module.__ptflops_backup_flops__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__ptflops_backup_params__'):
                module.__params__ = module.__ptflops_backup_params__


class torch_function_wrapper:
    def __init__(self, op, handler, collector) -> None:
        self.collector = collector
        self.op = op
        self.handler = handler

    def __call__(self, *args, **kwds):
        flops = self.handler(*args, **kwds)
        self.collector.append(flops)
        return self.op(*args, **kwds)


def patch_functional(collector):
    # F.linear = torch_function_wrapper(F.linear, FUNCTIONAL_MAPPING[F.linear], collector)
    F.relu = torch_function_wrapper(F.relu, FUNCTIONAL_MAPPING[F.relu], collector)
    F.prelu = torch_function_wrapper(F.prelu, FUNCTIONAL_MAPPING[F.prelu], collector)
    F.elu = torch_function_wrapper(F.elu, FUNCTIONAL_MAPPING[F.elu], collector)
    F.relu6 = torch_function_wrapper(F.relu6, FUNCTIONAL_MAPPING[F.relu6], collector)
    F.gelu = torch_function_wrapper(F.gelu, FUNCTIONAL_MAPPING[F.gelu], collector)

    F.avg_pool1d = torch_function_wrapper(F.avg_pool1d,
                                          FUNCTIONAL_MAPPING[F.avg_pool1d], collector)
    F.avg_pool2d = torch_function_wrapper(F.avg_pool2d,
                                          FUNCTIONAL_MAPPING[F.avg_pool2d], collector)
    F.avg_pool3d = torch_function_wrapper(F.avg_pool3d,
                                          FUNCTIONAL_MAPPING[F.avg_pool3d], collector)
    F.max_pool1d = torch_function_wrapper(F.max_pool1d,
                                          FUNCTIONAL_MAPPING[F.max_pool1d], collector)
    F.max_pool2d = torch_function_wrapper(F.max_pool2d,
                                          FUNCTIONAL_MAPPING[F.max_pool2d], collector)
    F.max_pool3d = torch_function_wrapper(F.max_pool3d,
                                          FUNCTIONAL_MAPPING[F.max_pool3d], collector)
    F.adaptive_avg_pool1d = torch_function_wrapper(
        F.adaptive_avg_pool1d, FUNCTIONAL_MAPPING[F.adaptive_avg_pool1d], collector)
    F.adaptive_avg_pool2d = torch_function_wrapper(
        F.adaptive_avg_pool2d, FUNCTIONAL_MAPPING[F.adaptive_avg_pool2d], collector)
    F.adaptive_avg_pool3d = torch_function_wrapper(
        F.adaptive_avg_pool3d, FUNCTIONAL_MAPPING[F.adaptive_avg_pool3d], collector)
    F.adaptive_max_pool1d = torch_function_wrapper(
        F.adaptive_max_pool1d, FUNCTIONAL_MAPPING[F.adaptive_max_pool1d], collector)
    F.adaptive_max_pool2d = torch_function_wrapper(
        F.adaptive_max_pool2d, FUNCTIONAL_MAPPING[F.adaptive_max_pool2d], collector)
    F.adaptive_max_pool3d = torch_function_wrapper(
        F.adaptive_max_pool3d, FUNCTIONAL_MAPPING[F.adaptive_max_pool3d], collector)

    F.softmax = torch_function_wrapper(
        F.softmax, FUNCTIONAL_MAPPING[F.softmax], collector)

    F.upsample = torch_function_wrapper(
        F.upsample, FUNCTIONAL_MAPPING[F.upsample], collector)
    F.interpolate = torch_function_wrapper(
        F.interpolate, FUNCTIONAL_MAPPING[F.interpolate], collector)

    if hasattr(F, "silu"):
        F.silu = torch_function_wrapper(F.silu, FUNCTIONAL_MAPPING[F.silu], collector)


def unpatch_functional():
    # F.linear = F.linear.op
    F.relu = F.relu.op
    F.prelu = F.prelu.op
    F.elu = F.elu.op
    F.relu6 = F.relu6.op
    F.gelu = F.gelu.op
    if hasattr(F, "silu"):
        F.silu = F.silu.op

    F.avg_pool1d = F.avg_pool1d.op
    F.avg_pool2d = F.avg_pool2d.op
    F.avg_pool3d = F.avg_pool3d.op
    F.max_pool1d = F.max_pool1d.op
    F.max_pool2d = F.max_pool2d.op
    F.max_pool3d = F.max_pool3d.op
    F.adaptive_avg_pool1d = F.adaptive_avg_pool1d.op
    F.adaptive_avg_pool2d = F.adaptive_avg_pool2d.op
    F.adaptive_avg_pool3d = F.adaptive_avg_pool3d.op
    F.adaptive_max_pool1d = F.adaptive_max_pool1d.op
    F.adaptive_max_pool2d = F.adaptive_max_pool2d.op
    F.adaptive_max_pool3d = F.adaptive_max_pool3d.op

    F.softmax = F.softmax.op

    F.upsample = F.upsample.op
    F.interpolate = F.interpolate.op


def wrap_tensor_op(op, collector):
    tensor_op_handler = torch_function_wrapper(
        op, TENSOR_OPS_MAPPING[op], collector)

    def wrapper(*args, **kwargs):
        return tensor_op_handler(*args, **kwargs)

    wrapper.op = tensor_op_handler.op

    return wrapper


def patch_tensor_ops(collector):
    torch.matmul = torch_function_wrapper(
        torch.matmul, TENSOR_OPS_MAPPING[torch.matmul], collector)
    torch.Tensor.matmul = wrap_tensor_op(torch.Tensor.matmul, collector)
    torch.mm = torch_function_wrapper(
        torch.mm, TENSOR_OPS_MAPPING[torch.mm], collector)
    torch.Tensor.mm = wrap_tensor_op(torch.Tensor.mm, collector)
    torch.bmm = torch_function_wrapper(
        torch.bmm, TENSOR_OPS_MAPPING[torch.bmm], collector)
    torch.Tensor.bmm = wrap_tensor_op(torch.Tensor.bmm, collector)

    torch.addmm = torch_function_wrapper(
        torch.addmm, TENSOR_OPS_MAPPING[torch.addmm], collector)
    torch.Tensor.addmm = wrap_tensor_op(torch.Tensor.addmm, collector)
    torch.baddbmm = torch_function_wrapper(
        torch.baddbmm, TENSOR_OPS_MAPPING[torch.baddbmm], collector)

    torch.mul = torch_function_wrapper(
        torch.mul, TENSOR_OPS_MAPPING[torch.mul], collector)
    torch.Tensor.mul = wrap_tensor_op(torch.Tensor.mul, collector)
    torch.add = torch_function_wrapper(
        torch.add, TENSOR_OPS_MAPPING[torch.add], collector)
    torch.Tensor.add = wrap_tensor_op(torch.Tensor.add, collector)


def unpatch_tensor_ops():
    torch.matmul = torch.matmul.op
    torch.Tensor.matmul = torch.Tensor.matmul.op
    torch.mm = torch.mm.op
    torch.Tensor.mm = torch.Tensor.mm.op
    torch.bmm = torch.bmm.op
    torch.Tensor.bmm = torch.Tensor.bmm.op

    torch.addmm = torch.addmm.op
    torch.Tensor.addmm = torch.Tensor.addmm.op
    torch.baddbmm = torch.baddbmm.op

    torch.mul = torch.mul.op
    torch.Tensor.mul = torch.Tensor.mul.op
    torch.add = torch.add.op
    torch.Tensor.add = torch.Tensor.add.op


# ---- Internal functions
def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    input_last_dim = input.shape[-1]
    pre_last_dims_prod = np.prod(input.shape[0:-1], dtype=np.int64)
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int((input_last_dim * output_last_dim + bias_flops)
                            * pre_last_dims_prod)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape, dtype=np.int64))


def bn_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape, dtype=np.int64)
    if hasattr(module, "affine") and module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def conv_flops_counter_hook(conv_module, input, output, extra_per_position_flops=0):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims, dtype=np.int64)) * \
        (in_channels * filters_per_channel + extra_per_position_flops)

    active_elements_count = batch_size * int(np.prod(output_dims, dtype=np.int64))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def deformable_conv_flops_counter_hook(conv_module, input, output):
    # 20 = 4 x 5 is an approximate cost of billinear interpolation, 2x2 grid is used
    # 4 is an approximate cost of fractional coordinates computation
    deformable_conv_extra_complexity = 20 + 4
    # consider also modulation multiplication
    if len(input) == 3 and input[2] is not None:
        deformable_conv_extra_complexity += 1
    conv_flops_counter_hook(conv_module, input, output, deformable_conv_extra_complexity)


def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size*3
        # last two hadamard product and add
        flops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_flops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_flops_counter_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    flops = 0

    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    flops = 0

    # Q scaling
    flops += qlen * qdim

    # Initial projections
    flops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_flops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += qlen * vdim * (vdim + 1)

    flops *= batch_size
    multihead_attention_module.__flops__ += int(flops)


def timm_attention_counter_hook(attention_module, input, output):
    flops = 0
    B, N, C = input[0].shape  # [Batch_size, Seq_len, Dimension]

    # QKV projection is already covered in MODULES_MAPPING

    # Q scaling
    flops += N * attention_module.head_dim * attention_module.num_heads

    # head flops
    head_flops = (
        (N * N * attention_module.head_dim)  # QK^T
        + (N * N)  # softmax
        + (N * N * attention_module.head_dim)  # AV
    )
    flops += head_flops * attention_module.num_heads

    # Final projection is already covered in MODULES_MAPPING

    flops *= B
    attention_module.__flops__ += int(flops)


CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_flops_counter_hook,
    nn.AvgPool1d: pool_flops_counter_hook,
    nn.AvgPool2d: pool_flops_counter_hook,
    nn.MaxPool2d: pool_flops_counter_hook,
    nn.MaxPool3d: pool_flops_counter_hook,
    nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_flops_counter_hook,
    nn.BatchNorm2d: bn_flops_counter_hook,
    nn.BatchNorm3d: bn_flops_counter_hook,

    nn.InstanceNorm1d: bn_flops_counter_hook,
    nn.InstanceNorm2d: bn_flops_counter_hook,
    nn.InstanceNorm3d: bn_flops_counter_hook,
    nn.GroupNorm: bn_flops_counter_hook,
    nn.LayerNorm: bn_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_flops_counter_hook,
    nn.ConvTranspose2d: conv_flops_counter_hook,
    nn.ConvTranspose3d: conv_flops_counter_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.RNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_flops_counter_hook

try:
    import torchvision.ops as tops
    MODULES_MAPPING[tops.DeformConv2d] = deformable_conv_flops_counter_hook
except ImportError:
    pass

try:
    from timm.models.vision_transformer import Attention as timm_Attention
    MODULES_MAPPING[timm_Attention] = timm_attention_counter_hook
except ImportError:
    pass


def _linear_functional_flops_hook(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = input.numel() * out_features
    if bias is not None:
        macs += out_features
    return macs


def _numel_functional_flops_hook(input, *args, **kwargs):
    return input.numel()


def _interpolate_functional_flops_hook(*args, **kwargs):
    input = kwargs.get('input', None)
    if input is None and len(args) > 0:
        input = args[0]

    size = kwargs.get('size', None)
    if size is None and len(args) > 1:
        size = args[1]

    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            return int(np.prod(size, dtype=np.int64))
        else:
            return int(size)

    scale_factor = kwargs.get('scale_factor', None)
    if scale_factor is None and len(args) > 2:
        scale_factor = args[2]
    assert scale_factor is not None, "either size or scale_factor"
    "should be passes to interpolate"

    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops *= int(np.prod(scale_factor, dtype=np.int64))
    else:
        flops *= scale_factor**len(input)

    return flops


def _matmul_tensor_flops_hook(input, other, *args, **kwargs):
    flops = np.prod(input.shape, dtype=np.int64) * other.shape[-1]
    return flops


def _addmm_tensor_flops_hook(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    flops = np.prod(mat1.shape, dtype=np.int64) * mat2.shape[-1]
    if beta != 0:
        flops += np.prod(input.shape, dtype=np.int64)
    return flops


def _elementwise_tensor_flops_hook(input, other, *args, **kwargs):
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return np.prod(other.shape, dtype=np.int64)
        else:
            return 1
    elif not torch.is_tensor(other):
        return np.prod(input.shape, dtype=np.int64)
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = np.prod(final_shape, dtype=np.int64)
        return flops


FUNCTIONAL_MAPPING = {
    F.linear: _linear_functional_flops_hook,
    F.relu: _numel_functional_flops_hook,
    F.prelu: _numel_functional_flops_hook,
    F.elu: _numel_functional_flops_hook,
    F.relu6: _numel_functional_flops_hook,
    F.gelu: _numel_functional_flops_hook,

    F.avg_pool1d: _numel_functional_flops_hook,
    F.avg_pool2d: _numel_functional_flops_hook,
    F.avg_pool3d: _numel_functional_flops_hook,
    F.max_pool1d: _numel_functional_flops_hook,
    F.max_pool2d: _numel_functional_flops_hook,
    F.max_pool3d: _numel_functional_flops_hook,
    F.adaptive_avg_pool1d: _numel_functional_flops_hook,
    F.adaptive_avg_pool2d: _numel_functional_flops_hook,
    F.adaptive_avg_pool3d: _numel_functional_flops_hook,
    F.adaptive_max_pool1d: _numel_functional_flops_hook,
    F.adaptive_max_pool2d: _numel_functional_flops_hook,
    F.adaptive_max_pool3d: _numel_functional_flops_hook,

    F.softmax: _numel_functional_flops_hook,

    F.upsample: _interpolate_functional_flops_hook,
    F.interpolate: _interpolate_functional_flops_hook,
}

if hasattr(F, "silu"):
    FUNCTIONAL_MAPPING[F.silu] = _numel_functional_flops_hook


TENSOR_OPS_MAPPING = {
    torch.matmul: _matmul_tensor_flops_hook,
    torch.Tensor.matmul: _matmul_tensor_flops_hook,
    torch.mm: _matmul_tensor_flops_hook,
    torch.Tensor.mm: _matmul_tensor_flops_hook,
    torch.bmm: _matmul_tensor_flops_hook,
    torch.Tensor.bmm: _matmul_tensor_flops_hook,

    torch.addmm: _addmm_tensor_flops_hook,
    torch.baddbmm: _addmm_tensor_flops_hook,
    torch.Tensor.addmm: _addmm_tensor_flops_hook,

    torch.mul: _elementwise_tensor_flops_hook,
    torch.Tensor.mul: _elementwise_tensor_flops_hook,
    torch.add: _elementwise_tensor_flops_hook,
    torch.Tensor.add: _elementwise_tensor_flops_hook,
}
