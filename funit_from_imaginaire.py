# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from functools import partial
from types import SimpleNamespace
import warnings
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import Upsample as NearestUpsample

class ApplyNoise(nn.Module):
    r"""Add Gaussian noise to the input tensor."""

    def __init__(self):
        super().__init__()
        # scale of the noise
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        r"""

        Args:
            x (tensor): Input tensor.
            noise (tensor, optional, default=``None``) : Noise tensor to be
                added to the input.
        """
        if noise is None:
            sz = x.size()
            noise = x.new_empty(sz[0], 1, *sz[2:]).normal_()

        return x + self.weight * noise

class _BaseConvBlock(nn.Module):
    r"""An abstract wrapper class that wraps a torch convolution or linear layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, order, input_dim):
        super().__init__()
        from .nonlinearity import get_nonlinearity_layer
        from .weight_norm import get_weight_norm_layer
        from .activation_norm import get_activation_norm_layer
        self.weight_norm_type = weight_norm_type

        # Convolutional layer.
        if weight_norm_params is None:
            weight_norm_params = SimpleNamespace()
        weight_norm = get_weight_norm_layer(
            weight_norm_type, **vars(weight_norm_params))
        conv_layer = weight_norm(self._get_conv_layer(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, input_dim))

        # Noise injection layer.
        noise_layer = ApplyNoise() if apply_noise else None

        # Normalization layer.
        conv_before_norm = order.find('C') < order.find('N')
        norm_channels = out_channels if conv_before_norm else in_channels
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace()
        activation_norm_layer = get_activation_norm_layer(
            norm_channels,
            activation_norm_type,
            input_dim,
            **vars(activation_norm_params))

        # Nonlinearity layer.
        nonlinearity_layer = get_nonlinearity_layer(
            nonlinearity, inplace=inplace_nonlinearity)

        # Mapping from operation names to layers.
        mappings = {'C': {'conv': conv_layer},
                    'N': {'norm': activation_norm_layer},
                    'A': {'nonlinearity': nonlinearity_layer}}

        # All layers in order.
        self.layers = nn.ModuleDict()
        for op in order:
            if list(mappings[op].values())[0] is not None:
                self.layers.update(mappings[op])
                if op == 'C' and noise_layer is not None:
                    # Inject noise after convolution.
                    self.layers.update({'noise': noise_layer})

        # Whether this block expects conditional inputs.
        self.conditional = \
            getattr(conv_layer, 'conditional', False) or \
            getattr(activation_norm_layer, 'conditional', False)

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        """
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                # Layers that require conditional inputs.
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            else:
                x = layer(x)
        return x

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        input_dim):
        # Returns the convolutional layer.
        if input_dim == 0:
            layer = nn.Linear(in_channels, out_channels, bias)
        else:
            layer_type = getattr(nn, 'Conv%dd' % input_dim)
            layer = layer_type(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias, padding_mode)
        return layer

    def __repr__(self):
        main_str = self._get_name() + '('
        child_lines = []
        for name, layer in self.layers.items():
            mod_str = repr(layer)
            if name == 'conv' and self.weight_norm_type != 'none' and \
                    self.weight_norm_type != '':
                mod_str = mod_str[:-1] + \
                    ', weight_norm={}'.format(self.weight_norm_type) + ')'
            mod_str = self._addindent(mod_str, 2)
            child_lines.append(mod_str)
        if len(child_lines) == 1:
            main_str += child_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(child_lines) + '\n'

        main_str += ')'
        return main_str

    @staticmethod
    def _addindent(s_, numSpaces):
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

class Conv2dBlock(_BaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Conv2d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, order, 2)

class LinearBlock(_BaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Linear`` with normalization and
    nonlinearity.

    Args:
        in_features (int): Number of channels in the input tensor.
        out_features (int): Number of channels in the output tensor.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, add
            Gaussian noise with learnable magnitude after the
            fully-connected layer.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: fully-connected,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_features, out_features, bias=True,
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_features, out_features, None, None,
                         None, None, None, bias,
                         None, weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         order, 0)

class _BaseResBlock(nn.Module):
    r"""An abstract class for residual blocks.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity, apply_noise,
                 hidden_channels_equal_out_channels,
                 order, block, learn_shortcut):
        super().__init__()
        if order == 'pre_act':
            order = 'NACNAC'
        if isinstance(bias, bool):
            # The bias for conv_block_0, conv_block_1, and conv_block_s.
            biases = [bias, bias, bias]
        elif isinstance(bias, list):
            if len(bias) == 3:
                biases = bias
            else:
                raise ValueError('Bias list must be 3.')
        else:
            raise ValueError('Bias must be either an integer or s list.')
        self.learn_shortcut = (in_channels != out_channels) or learn_shortcut
        if len(order) > 6 or len(order) < 5:
            raise ValueError('order must be either 5 or 6 characters')
        if hidden_channels_equal_out_channels:
            hidden_channels = out_channels
        else:
            hidden_channels = min(in_channels, out_channels)

        # Parameters that are specific for convolutions.
        conv_main_params = {}
        conv_skip_params = {}
        if block != LinearBlock:
            conv_base_params = dict(stride=1, dilation=dilation,
                                    groups=groups, padding_mode=padding_mode)
            conv_main_params.update(conv_base_params)
            conv_main_params.update(
                dict(kernel_size=kernel_size,
                     activation_norm_type=activation_norm_type,
                     activation_norm_params=activation_norm_params,
                     padding=padding))
            conv_skip_params.update(conv_base_params)
            conv_skip_params.update(dict(kernel_size=1))
            if skip_activation_norm:
                conv_skip_params.update(
                    dict(activation_norm_type=activation_norm_type,
                         activation_norm_params=activation_norm_params))

        # Other parameters.
        other_params = dict(weight_norm_type=weight_norm_type,
                            weight_norm_params=weight_norm_params,
                            apply_noise=apply_noise)

        # Residual branch.
        if order.find('A') < order.find('C') and \
                (activation_norm_type == '' or activation_norm_type == 'none'):
            # Nonlinearity is the first operation in the residual path.
            # In-place nonlinearity will modify the input variable and cause
            # backward error.
            first_inplace = False
        else:
            first_inplace = inplace_nonlinearity
        self.conv_block_0 = block(in_channels, hidden_channels,
                                  bias=biases[0],
                                  nonlinearity=nonlinearity,
                                  order=order[0:3],
                                  inplace_nonlinearity=first_inplace,
                                  **conv_main_params,
                                  **other_params)
        self.conv_block_1 = block(hidden_channels, out_channels,
                                  bias=biases[1],
                                  nonlinearity=nonlinearity,
                                  order=order[3:],
                                  inplace_nonlinearity=inplace_nonlinearity,
                                  **conv_main_params,
                                  **other_params)

        # Shortcut branch.
        if self.learn_shortcut:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = block(in_channels, out_channels,
                                      bias=biases[2],
                                      nonlinearity=skip_nonlinearity_type,
                                      order=order[0:3],
                                      **conv_skip_params,
                                      **other_params)

        # Whether this block expects conditional inputs.
        self.conditional = \
            getattr(self.conv_block_0, 'conditional', False) or \
            getattr(self.conv_block_1, 'conditional', False)

    def conv_blocks(self, x, *cond_inputs, **kw_cond_inputs):
        r"""Returns the output of the residual branch.

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            dx (tensor): Output tensor.
        """
        dx = self.conv_block_0(x, *cond_inputs, **kw_cond_inputs)
        dx = self.conv_block_1(dx, *cond_inputs, **kw_cond_inputs)
        return dx

    def forward(self, x, *cond_inputs, do_checkpoint=False, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            do_checkpoint (bool, optional, default=``False``) If ``True``,
                trade compute for memory by checkpointing the model.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            output (tensor): Output tensor.
        """
        if do_checkpoint:
            dx = checkpoint(self.conv_blocks, x, *cond_inputs, **kw_cond_inputs)
        else:
            dx = self.conv_blocks(x, *cond_inputs, **kw_cond_inputs)

        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs, **kw_cond_inputs)
        else:
            x_shortcut = x
        output = x_shortcut + dx
        return output

class Res2dBlock(_BaseResBlock):
    r"""Residual block for 2D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, Conv2dBlock, learn_shortcut)

class _BaseUpResBlock(_BaseResBlock):
    r"""An abstract class for residual blocks with upsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, hidden_channels_equal_out_channels,
                 order, block, upsample, up_factor, learn_shortcut):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)
        self.order = order
        self.upsample = upsample(scale_factor=up_factor)

    def forward(self, x, *cond_inputs):
        r"""Implementation of the up residual block forward function.
        If the order is 'NAC' for the first residual block, we will first
        do the activation norm and nonlinearity, in the original resolution.
        We will then upsample the activation map to a higher resolution. We
        then do the convolution.
        It is is other orders, then we first do the whole processing and
        then upsample.

        Args:
            x (tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional input.
        Returns:
            output (tensor) : Output tensor.
        """
        # In this particular upsample residual block operation, we first
        # upsample the skip connection.
        if self.learn_shortcut:
            x_shortcut = self.upsample(x)
            x_shortcut = self.conv_block_s(x_shortcut, *cond_inputs)
        else:
            x_shortcut = self.upsample(x)

        if self.order[0:3] == 'NAC':
            for ix, layer in enumerate(self.conv_block_0.layers.values()):
                if getattr(layer, 'conditional', False):
                    x = layer(x, *cond_inputs)
                else:
                    x = layer(x)
                if ix == 1:
                    x = self.upsample(x)
        else:
            x = self.conv_block_0(x, *cond_inputs)
            x = self.upsample(x)
        x = self.conv_block_1(x, *cond_inputs)

        output = x_shortcut + x
        return output

class UpRes2dBlock(_BaseUpResBlock):
    r"""Residual block for 2D input with downsampling.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        upsample (class, optional, default=NearestUpsample): PPytorch
            upsampling layer to be used.
        up_factor (int, optional, default=2): Upsampling factor.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', upsample=NearestUpsample, up_factor=2,
                 learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, Conv2dBlock,
                         upsample, up_factor, learn_shortcut)

class Decoder(nn.Module):
    r"""Improved FUNIT decoder.

    Args:
        num_enc_output_channels (int): Number of content feature channels.
        style_channels (int): Dimension of the style code.
        num_image_channels (int): Number of image channels.
        num_upsamples (int): How many times we are going to apply
            upsample residual block.
    """

    def __init__(self,
                 num_enc_output_channels,
                 style_channels,
                 num_image_channels=3,
                 num_upsamples=4,
                 padding_type='reflect',
                 weight_norm_type='none',
                 nonlinearity='relu'):
        super(Decoder, self).__init__()
        adain_params = SimpleNamespace(
            activation_norm_type='instance',
            activation_norm_params=SimpleNamespace(affine=False),
            cond_dims=style_channels)

        base_res_block = partial(Res2dBlock,
                                 kernel_size=3,
                                 padding=1,
                                 padding_mode=padding_type,
                                 nonlinearity=nonlinearity,
                                 activation_norm_type='adaptive',
                                 activation_norm_params=adain_params,
                                 weight_norm_type=weight_norm_type)

        base_up_res_block = partial(UpRes2dBlock,
                                    kernel_size=5,
                                    padding=2,
                                    padding_mode=padding_type,
                                    weight_norm_type=weight_norm_type,
                                    activation_norm_type='adaptive',
                                    activation_norm_params=adain_params,
                                    skip_activation_norm='instance',
                                    skip_nonlinearity=nonlinearity,
                                    nonlinearity=nonlinearity,
                                    hidden_channels_equal_out_channels=True)

        dims = num_enc_output_channels

        # Residual blocks with AdaIN.
        self.decoder = nn.ModuleList()
        self.decoder += [base_res_block(dims, dims)]
        self.decoder += [base_res_block(dims, dims)]
        for _ in range(num_upsamples):
            self.decoder += [base_up_res_block(dims, dims // 2)]
            dims = dims // 2
        self.decoder += [Conv2dBlock(dims,
                                     num_image_channels,
                                     kernel_size=7,
                                     stride=1,
                                     padding=3,
                                     padding_mode='reflect',
                                     nonlinearity='tanh')]

    def forward(self, x, style):
        r"""

        Args:
            x (tensor): Content embedding of the content image.
            style (tensor): Style embedding of the style image.
        """
        for block in self.decoder:
            if getattr(block, 'conditional', False):
                x = block(x, style)
            else:
                x = block(x)
        return x

class StyleEncoder(nn.Module):
    r"""Improved FUNIT Style Encoder. This is basically the same as the
    original FUNIT Style Encoder.

    Args:
        num_downsamples (int): Number of times we reduce resolution by
            2x2.
        image_channels (int): Number of input image channels.
        num_filters (int): Base filter number.
        style_channels (int): Style code dimension.
        padding_mode (str): Padding mode.
        activation_norm_type (str): Type of activation normalization.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        nonlinearity (str): Nonlinearity.
    """

    def __init__(self,
                 num_downsamples,
                 image_channels,
                 num_filters,
                 style_channels,
                 padding_mode,
                 activation_norm_type,
                 weight_norm_type,
                 nonlinearity):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type=activation_norm_type,
                           weight_norm_type=weight_norm_type,
                           nonlinearity=nonlinearity,
                           inplace_nonlinearity=True)
        model = []
        model += [Conv2dBlock(image_channels, num_filters, 7, 1, 3,
                              **conv_params)]
        for i in range(2):
            model += [Conv2dBlock(num_filters, 2 * num_filters, 4, 2, 1,
                                  **conv_params)]
            num_filters *= 2
        for i in range(num_downsamples - 2):
            model += [Conv2dBlock(num_filters, num_filters, 4, 2, 1,
                                  **conv_params)]
        model += [nn.AdaptiveAvgPool2d(1)]
        model += [nn.Conv2d(num_filters, style_channels, 1, 1, 0)]
        self.model = nn.Sequential(*model)
        self.output_dim = num_filters

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input image.
        """
        return self.model(x)

class ContentEncoder(nn.Module):
    r"""Improved FUNIT Content Encoder. This is basically the same as the
    original FUNIT content encoder.

    Args:
        num_downsamples (int): Number of times we reduce resolution by
           2x2.
        num_res_blocks (int): Number of times we append residual block
           after all the downsampling modules.
        image_channels (int): Number of input image channels.
        num_filters (int): Base filter number.
        padding_mode (str): Padding mode
        activation_norm_type (str): Type of activation normalization.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
        nonlinearity (str): Nonlinearity.
    """

    def __init__(self,
                 num_downsamples,
                 num_res_blocks,
                 image_channels,
                 num_filters,
                 padding_mode,
                 activation_norm_type,
                 weight_norm_type,
                 nonlinearity):
        super().__init__()
        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type=activation_norm_type,
                           weight_norm_type=weight_norm_type,
                           nonlinearity=nonlinearity,
                           inplace_nonlinearity=True,
                           order='CNACNA')
        model = []
        model += [Conv2dBlock(image_channels, num_filters, 7, 1, 3,
                              **conv_params)]
        dims = num_filters
        for i in range(num_downsamples):
            model += [Conv2dBlock(dims, dims * 2, 4, 2, 1, **conv_params)]
            dims *= 2

        for _ in range(num_res_blocks):
            model += [Res2dBlock(dims, dims, **conv_params)]
        self.model = nn.Sequential(*model)
        self.output_dim = dims

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input image.
        """
        return self.model(x)

class Discriminator(nn.Module):
    r"""Discriminator in the improved FUNIT baseline in the COCO-FUNIT paper.

    Args:
        dis_cfg (obj): Discriminator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, dis_cfg, data_cfg):
        super().__init__()
        self.model = ResDiscriminator(**vars(dis_cfg))

    def forward(self, data, net_G_output, recon=True):
        r"""Improved FUNIT discriminator forward function.

        Args:
            data (dict): Training data at the current iteration.
            net_G_output (dict): Fake data generated at the current iteration.
            recon (bool): If ``True``, also classifies reconstructed images.
        """
        source_labels = data['labels_content']
        target_labels = data['labels_style']
        fake_out_trans, fake_features_trans = \
            self.model(net_G_output['images_trans'], target_labels)
        output = dict(fake_out_trans=fake_out_trans,
                      fake_features_trans=fake_features_trans)

        real_out_style, real_features_style = \
            self.model(data['images_style'], target_labels)
        output.update(dict(real_out_style=real_out_style,
                           real_features_style=real_features_style))
        if recon:
            fake_out_recon, fake_features_recon = \
                self.model(net_G_output['images_recon'], source_labels)
            output.update(dict(fake_out_recon=fake_out_recon,
                               fake_features_recon=fake_features_recon))
        return output

class ResDiscriminator(nn.Module):
    r"""Residual discriminator architecture used in the FUNIT paper."""

    def __init__(self,
                 image_channels=3,
                 num_classes=119,
                 num_filters=64,
                 max_num_filters=1024,
                 num_layers=6,
                 padding_mode='reflect',
                 weight_norm_type='',
                 **kwargs):
        super().__init__()
        for key in kwargs:
            if key != 'type':
                warnings.warn(
                    "Discriminator argument {} is not used".format(key))

        conv_params = dict(padding_mode=padding_mode,
                           activation_norm_type='none',
                           weight_norm_type=weight_norm_type,
                           bias=[True, True, True],
                           nonlinearity='leakyrelu',
                           order='NACNAC')

        first_kernel_size = 7
        first_padding = (first_kernel_size - 1) // 2
        model = [Conv2dBlock(image_channels, num_filters,
                             first_kernel_size, 1, first_padding,
                             padding_mode=padding_mode,
                             weight_norm_type=weight_norm_type)]
        for i in range(num_layers):
            num_filters_prev = num_filters
            num_filters = min(num_filters * 2, max_num_filters)
            model += [Res2dBlock(num_filters_prev, num_filters_prev,
                                 **conv_params),
                      Res2dBlock(num_filters_prev, num_filters,
                                 **conv_params)]
            if i != num_layers - 1:
                model += [nn.ReflectionPad2d(1),
                          nn.AvgPool2d(3, stride=2)]
        self.model = nn.Sequential(*model)
        self.classifier = Conv2dBlock(num_filters, 1, 1, 1, 0,
                                      nonlinearity='leakyrelu',
                                      weight_norm_type=weight_norm_type,
                                      order='NACNAC')

        self.embedder = nn.Embedding(num_classes, num_filters)

    def forward(self, images, labels=None):
        r"""Forward function of the projection discriminator.

        Args:
            images (image tensor): Images inputted to the discriminator.
            labels (long int tensor): Class labels of the images.
        """
        assert (images.size(0) == labels.size(0))
        features = self.model(images)
        outputs = self.classifier(features)
        features_1x1 = features.mean(3).mean(2)
        if labels is None:
            return features_1x1
        embeddings = self.embedder(labels)
        outputs += torch.sum(embeddings * features_1x1, dim=1,
                             keepdim=True).view(images.size(0), 1, 1, 1)
        return outputs, features_1x1
