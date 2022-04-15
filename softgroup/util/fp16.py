# Simplfied from mmcv.
# Directly use torch.cuda.amp.autocast for mix-precision and support sparse tensor
import functools
from collections import abc
from inspect import getfullargspec

import spconv.pytorch as spconv
import torch


def cast_tensor_type(inputs, src_type, dst_type):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type) if inputs.dtype == src_type else inputs
    elif isinstance(inputs, spconv.SparseConvTensor):
        if inputs.features.dtype == src_type:
            features = inputs.features.to(dst_type)
            inputs = inputs.replace_feature(features)
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({k: cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()})
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def force_fp32(apply_to=None, out_fp16=False):

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the '
                                'method of nn.Module')
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            with torch.cuda.amp.autocast(enabled=False):
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper
