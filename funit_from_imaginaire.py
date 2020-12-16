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
import functools
from torch.nn import functional as F
from torch.nn import SyncBatchNorm
from torch.nn.utils import spectral_norm, weight_norm

class PartialSequential(nn.Sequential):
    r"""Sequential block for partial convolutions."""
    def __init__(self, *modules):
        super(PartialSequential, self).__init__(*modules)

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input tensor.
        """
        act = x[:, :-1]
        mask = x[:, -1].unsqueeze(1)
        for module in self:
            act, mask = module(act, mask_in=mask)
        return act

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

class _BasePartialConvBlock(_BaseConvBlock):
    r"""An abstract wrapper class that wraps a partial convolutional layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity,
                 multi_channel, return_mask,
                 apply_noise, order, input_dim):
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self.partial_conv = True
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         order, input_dim)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        input_dim):
        if input_dim == 2:
            layer_type = PartialConv2d
        elif input_dim == 3:
            layer_type = PartialConv3d
        else:
            raise ValueError('Partial conv only supports 2D and 3D conv now.')
        layer = layer_type(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode,
            multi_channel=self.multi_channel, return_mask=self.return_mask)
        return layer

    def forward(self, x, *cond_inputs, mask_in=None, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            mask_in (tensor, optional, default=``None``) If not ``None``,
                it masks the valid input region.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            (tuple):
              - x (tensor): Output tensor.
              - mask_out (tensor, optional): Masks the valid output region.
        """
        mask_out = None
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            elif getattr(layer, 'partial_conv', False):
                x = layer(x, mask_in=mask_in, **kw_cond_inputs)
                if type(x) == tuple:
                    x, mask_out = x
            else:
                x = layer(x)

        if mask_out is not None:
            return x, mask_out
        return x

class PartialConv2d(nn.Conv2d):
    r"""Partial 2D convolution in
    "Image inpainting for irregular holes using partial convolutions."
    Liu et al., ECCV 2018
    """

    def __init__(self, *args, multi_channel=False, return_mask=True, **kwargs):
        # whether the mask is multi-channel or not
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels,
                                                 self.in_channels,
                                                 self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0],
                                                 self.kernel_size[1])

        shape = self.weight_maskUpdater.shape
        self.slide_winsize = shape[1] * shape[2] * shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None
        self.partial_conv = True

    def forward(self, x, mask_in=None):
        r"""

        Args:
            x (tensor): Input tensor.
            mask_in (tensor, optional, default=``None``) If not ``None``,
                it masks the valid input region.
        """
        assert len(x.shape) == 4
        if mask_in is not None or self.last_size != tuple(x.shape):
            self.last_size = tuple(x.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)

                if mask_in is None:
                    # If mask is not provided, create a mask.
                    if self.multi_channel:
                        mask = torch.ones(x.data.shape[0],
                                          x.data.shape[1],
                                          x.data.shape[2],
                                          x.data.shape[3]).to(x)
                    else:
                        mask = torch.ones(1, 1, x.data.shape[2],
                                          x.data.shape[3]).to(x)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater,
                                            bias=None, stride=self.stride,
                                            padding=self.padding,
                                            dilation=self.dilation, groups=1)

                # For mixed precision training, eps from 1e-8 to 1e-6.
                eps = 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + eps)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(x, mask) if mask_in is not None else x)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class PartialConv3d(nn.Conv3d):
    r"""Partial 3D convolution in
    "Image inpainting for irregular holes using partial convolutions."
    Liu et al., ECCV 2018
    """

    def __init__(self, *args, multi_channel=False, return_mask=True, **kwargs):
        # whether the mask is multi-channel or not
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super(PartialConv3d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = \
                torch.ones(self.out_channels, self.in_channels,
                           self.kernel_size[0], self.kernel_size[1],
                           self.kernel_size[2])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0],
                                                 self.kernel_size[1],
                                                 self.kernel_size[2])
        self.weight_maskUpdater = self.weight_maskUpdater.to('cuda')

        shape = self.weight_maskUpdater.shape
        self.slide_winsize = shape[1] * shape[2] * shape[3] * shape[4]
        self.partial_conv = True

    def forward(self, x, mask_in=None):
        r"""

        Args:
            x (tensor): Input tensor.
            mask_in (tensor, optional, default=``None``) If not ``None``, it
                masks the valid input region.
        """
        assert len(x.shape) == 5

        with torch.no_grad():
            mask = mask_in
            update_mask = F.conv3d(mask, self.weight_maskUpdater, bias=None,
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=1)

            mask_ratio = self.slide_winsize / (update_mask + 1e-8)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio, update_mask)

        raw_out = super(PartialConv3d, self).forward(torch.mul(x, mask_in))

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, mask_ratio) + bias_view
            if mask_in is not None:
                output = torch.mul(output, update_mask)
        else:
            output = torch.mul(raw_out, mask_ratio)

        if self.return_mask:
            return output, update_mask
        else:
            return output

class PartialConv2dBlock(_BasePartialConvBlock):
    r"""A Wrapper class that wraps ``PartialConv2d`` with normalization and
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
        multi_channel (bool, optional, default=False): If ``True``, use
            different masks for different channels.
        return_mask (bool, optional, default=True): If ``True``, the
            forward call also returns a new mask.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 multi_channel=False, return_mask=True,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         multi_channel, return_mask, apply_noise, order, 2)

class HyperConv2d(nn.Module):
    r"""Hyper Conv2d initialization.

    Args:
        in_channels (int): Dummy parameter.
        out_channels (int): Dummy parameter.
        kernel_size (int or tuple): Dummy parameter.
        stride (int or tuple, optional, default=1):
            Stride of the convolution. Default: 1
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        padding_mode (string, optional, default='zeros'):
            ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True): If ``True``,
            adds a learnable bias to the output.
    """

    def __init__(self, in_channels=0, out_channels=0, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self.conditional = True

    def forward(self, x, *args, conv_weights=(None, None), **kwargs):
        r"""Hyper Conv2d forward. Convolve x using the provided weight and bias.

        Args:
            x (N x C x H x W tensor): Input tensor.
            conv_weights (N x C2 x C1 x k x k tensor or list of tensors):
                Convolution weights or [weight, bias].
        Returns:
            y (N x C2 x H x W tensor): Output tensor.
        """
        if conv_weights is None:
            conv_weight, conv_bias = None, None
        elif isinstance(conv_weights, torch.Tensor):
            conv_weight, conv_bias = conv_weights, None
        else:
            conv_weight, conv_bias = conv_weights

        if conv_weight is None:
            return x
        if conv_bias is None:
            if self.use_bias:
                raise ValueError('bias not provided but set to true during '
                                 'initialization')
            conv_bias = [None] * x.size(0)
        if self.padding_mode != 'zeros':
            x = F.pad(x, [self.padding] * 4, mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding

        y = None
        for i in range(x.size(0)):
            if self.stride >= 1:
                yi = F.conv2d(x[i: i + 1],
                              weight=conv_weight[i], bias=conv_bias[i],
                              stride=self.stride, padding=padding,
                              dilation=self.dilation, groups=self.groups)
            else:
                yi = F.conv_transpose2d(x[i: i + 1], weight=conv_weight[i],
                                        bias=conv_bias[i], padding=self.padding,
                                        stride=int(1 / self.stride),
                                        dilation=self.dilation,
                                        output_padding=self.padding,
                                        groups=self.groups)
            y = torch.cat([y, yi]) if y is not None else yi
        return y

class AdaptiveNorm(nn.Module):
    r"""Adaptive normalization layer. The layer first normalizes the input, then
    performs an affine transformation using parameters computed from the
    conditional inputs.

    Args:
        num_features (int): Number of channels in the input tensor.
        cond_dims (int): Number of channels in the conditional inputs.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``, or ``'weight_demod'``.
        projection (bool): If ``True``, project the conditional input to gamma
            and beta using a fully connected layer, otherwise directly use
            the conditional input as gamma and beta.
        separate_projection (bool): If ``True``, we will use two different
            layers for gamma and beta. Otherwise, we will use one layer. It
            matters only if you apply any weight norms to this layer.
        input_dim (int): Number of dimensions of the input tensor.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
    """

    def __init__(self, num_features, cond_dims, weight_norm_type='',
                 projection=True,
                 separate_projection=False,
                 input_dim=2,
                 activation_norm_type='instance',
                 activation_norm_params=None):
        super().__init__()
        self.projection = projection
        self.separate_projection = separate_projection
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              input_dim,
                                              **vars(activation_norm_params))
        if self.projection:
            if self.separate_projection:
                self.fc_gamma = \
                    LinearBlock(cond_dims, num_features,
                                weight_norm_type=weight_norm_type)
                self.fc_beta = \
                    LinearBlock(cond_dims, num_features,
                                weight_norm_type=weight_norm_type)
            else:
                self.fc = LinearBlock(cond_dims, num_features * 2,
                                      weight_norm_type=weight_norm_type)

        self.conditional = True

    def forward(self, x, y, **kwargs):
        r"""Adaptive Normalization forward.

        Args:
            x (N x C1 x * tensor): Input tensor.
            y (N x C2 tensor): Conditional information.
        Returns:
            out (N x C1 x * tensor): Output tensor.
        """
        if self.projection:
            if self.separate_projection:
                gamma = self.fc_gamma(y)
                beta = self.fc_beta(y)
                for _ in range(x.dim() - gamma.dim()):
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)
            else:
                y = self.fc(y)
                for _ in range(x.dim() - y.dim()):
                    y = y.unsqueeze(-1)
                gamma, beta = y.chunk(2, 1)
        else:
            for _ in range(x.dim() - y.dim()):
                y = y.unsqueeze(-1)
            gamma, beta = y.chunk(2, 1)
        x = self.norm(x) if self.norm is not None else x
        out = x * (1 + gamma) + beta
        return out

class SpatiallyAdaptiveNorm(nn.Module):
    r"""Spatially Adaptive Normalization (SPADE) initialization.

    Args:
        num_features (int) : Number of channels in the input tensor.
        cond_dims (int or list of int) : List of numbers of channels
            in the input.
        num_filters (int): Number of filters in SPADE.
        kernel_size (int): Kernel size of the convolutional filters in
            the SPADE layer.
         weight_norm_type (str): Type of weight normalization.
             ``'none'``, ``'spectral'``, or ``'weight'``.
        separate_projection (bool): If ``True``, we will use two different
            layers for gamma and beta. Otherwise, we will use one layer. It
            matters only if you apply any weight norms to this layer.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
    """

    def __init__(self,
                 num_features,
                 cond_dims,
                 num_filters=128,
                 kernel_size=3,
                 weight_norm_type='',
                 separate_projection=False,
                 activation_norm_type='sync_batch',
                 activation_norm_params=None,
                 partial=False):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        padding = kernel_size // 2
        self.separate_projection = separate_projection
        self.mlps = nn.ModuleList()
        self.gammas = nn.ModuleList()
        self.betas = nn.ModuleList()

        # Make cond_dims a list.
        if type(cond_dims) != list:
            cond_dims = [cond_dims]

        # Make num_filters a list.
        if not isinstance(num_filters, list):
            num_filters = [num_filters] * len(cond_dims)
        else:
            assert len(num_filters) >= len(cond_dims)

        # Make partial a list.
        if not isinstance(partial, list):
            partial = [partial] * len(cond_dims)
        else:
            assert len(partial) >= len(cond_dims)

        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            conv_block = PartialConv2dBlock if partial[i] else Conv2dBlock
            sequential = PartialSequential if partial[i] else nn.Sequential

            if num_filters[i] > 0:
                mlp += [conv_block(cond_dim,
                                   num_filters[i],
                                   kernel_size,
                                   padding=padding,
                                   weight_norm_type=weight_norm_type,
                                   nonlinearity='relu')]
            mlp_ch = cond_dim if num_filters[i] == 0 else num_filters[i]

            if self.separate_projection:
                if partial[i]:
                    raise NotImplementedError(
                        'Separate projection not yet implemented for ' +
                        'partial conv')
                self.mlps.append(nn.Sequential(*mlp))
                self.gammas.append(
                    conv_block(mlp_ch, num_features,
                               kernel_size,
                               padding=padding,
                               weight_norm_type=weight_norm_type))
                self.betas.append(
                    conv_block(mlp_ch, num_features,
                               kernel_size,
                               padding=padding,
                               weight_norm_type=weight_norm_type))
            else:
                mlp += [conv_block(mlp_ch, num_features * 2, kernel_size,
                                   padding=padding,
                                   weight_norm_type=weight_norm_type)]
                self.mlps.append(sequential(*mlp))

        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              2,
                                              **vars(activation_norm_params))
        self.conditional = True

    def forward(self, x, *cond_inputs, **kwargs):
        r"""Spatially Adaptive Normalization (SPADE) forward.

        Args:
            x (N x C1 x H x W tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional maps for SPADE.
        Returns:
            output (4D tensor) : Output tensor.
        """
        output = self.norm(x) if self.norm is not None else x
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            label_map = F.interpolate(cond_inputs[i], size=x.size()[2:],
                                      mode='nearest')
            if self.separate_projection:
                hidden = self.mlps[i](label_map)
                gamma = self.gammas[i](hidden)
                beta = self.betas[i](hidden)
            else:
                affine_params = self.mlps[i](label_map)
                gamma, beta = affine_params.chunk(2, dim=1)
            output = output * (1 + gamma) + beta
        return output

class HyperSpatiallyAdaptiveNorm(nn.Module):
    r"""Spatially Adaptive Normalization (SPADE) initialization.

    Args:
        num_features (int) : Number of channels in the input tensor.
        cond_dims (int or list of int) : List of numbers of channels
            in the conditional input.
        num_filters (int): Number of filters in SPADE.
        kernel_size (int): Kernel size of the convolutional filters in
            the SPADE layer.
         weight_norm_type (str): Type of weight normalization.
             ``'none'``, ``'spectral'``, or ``'weight'``.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``.
        is_hyper (bool): Whether to use hyper SPADE.
    """

    def __init__(self, num_features, cond_dims,
                 num_filters=0, kernel_size=3,
                 weight_norm_type='',
                 activation_norm_type='sync_batch', is_hyper=True):
        super().__init__()
        padding = kernel_size // 2
        self.mlps = nn.ModuleList()
        if type(cond_dims) != list:
            cond_dims = [cond_dims]

        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            if not is_hyper or (i != 0):
                if num_filters > 0:
                    mlp += [Conv2dBlock(cond_dim, num_filters, kernel_size,
                                        padding=padding,
                                        weight_norm_type=weight_norm_type,
                                        nonlinearity='relu')]
                mlp_ch = cond_dim if num_filters == 0 else num_filters
                mlp += [Conv2dBlock(mlp_ch, num_features * 2, kernel_size,
                                    padding=padding,
                                    weight_norm_type=weight_norm_type)]
                mlp = nn.Sequential(*mlp)
            else:
                if num_filters > 0:
                    raise ValueError('Multi hyper layer not supported yet.')
                mlp = HyperConv2d(padding=padding)
            self.mlps.append(mlp)

        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              2,
                                              affine=False)

        self.conditional = True

    def forward(self, x, *cond_inputs,
                norm_weights=(None, None), **kwargs):
        r"""Spatially Adaptive Normalization (SPADE) forward.

        Args:
            x (4D tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional maps for SPADE.
            norm_weights (5D tensor or list of tensors): conv weights or
            [weights, biases].
        Returns:
            output (4D tensor) : Output tensor.
        """
        output = self.norm(x)
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            if type(cond_inputs[i]) == list:
                cond_input, mask = cond_inputs[i]
                mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear',
                                     align_corners=False)
            else:
                cond_input = cond_inputs[i]
                mask = None
            label_map = F.interpolate(cond_input, size=x.size()[2:])
            if norm_weights is None or norm_weights[0] is None or i != 0:
                affine_params = self.mlps[i](label_map)
            else:
                affine_params = self.mlps[i](label_map,
                                             conv_weights=norm_weights)
            gamma, beta = affine_params.chunk(2, dim=1)
            if mask is not None:
                gamma = gamma * (1 - mask)
                beta = beta * (1 - mask)
            output = output * (1 + gamma) + beta
        return output

class LayerNorm2d(nn.Module):
    r"""Layer Normalization as introduced in
    https://arxiv.org/abs/1607.06450.
    This is the usual way to apply layer normalization in CNNs.
    Note that unlike the pytorch implementation which applies per-element
    scale and bias, here it applies per-channel scale and bias, similar to
    batch/instance normalization.

    Args:
        num_features (int): Number of channels in the input tensor.
        eps (float, optional, default=1e-5): a value added to the
            denominator for numerical stability.
        affine (bool, optional, default=False): If ``True``, performs
            affine transformation after normalization.
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input tensor.
        """
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def get_activation_norm_layer(num_features, norm_type,
                              input_dim, **norm_params):
    r"""Return an activation normalization layer.

    Args:
        num_features (int): Number of feature channels.
        norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        input_dim (int): Number of input dimensions.
        norm_params: Arbitrary keyword arguments that will be used to
            initialize the activation normalization.
    """
    input_dim = max(input_dim, 1)  # Norm1d works with both 0d and 1d inputs

    if norm_type == 'none' or norm_type == '':
        norm_layer = None
    elif norm_type == 'batch':
        norm = getattr(nn, 'BatchNorm%dd' % input_dim)
        norm_layer = norm(num_features, **norm_params)
    elif norm_type == 'instance':
        affine = norm_params.pop('affine', True)  # Use affine=True by default
        norm = getattr(nn, 'InstanceNorm%dd' % input_dim)
        norm_layer = norm(num_features, affine=affine, **norm_params)
    elif norm_type == 'sync_batch':
        # There is a bug of using amp O1 with synchronize batch norm.
        # The lines below fix it.
        affine = norm_params.pop('affine', True)
        # Always call SyncBN with affine=True
        norm_layer = SyncBatchNorm(num_features, affine=True, **norm_params)
        norm_layer.weight.requires_grad = affine
        norm_layer.bias.requires_grad = affine
    elif norm_type == 'layer':
        norm_layer = nn.LayerNorm(num_features, **norm_params)
    elif norm_type == 'layer_2d':
        norm_layer = LayerNorm2d(num_features, **norm_params)
    elif norm_type == 'group':
        norm_layer = nn.GroupNorm(num_channels=num_features, **norm_params)
    elif norm_type == 'adaptive':
        norm_layer = AdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'spatially_adaptive':
        if input_dim != 2:
            raise ValueError('Spatially adaptive normalization layers '
                             'only supports 2D input')
        norm_layer = SpatiallyAdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'hyper_spatially_adaptive':
        if input_dim != 2:
            raise ValueError('Spatially adaptive normalization layers '
                             'only supports 2D input')
        norm_layer = HyperSpatiallyAdaptiveNorm(num_features, **norm_params)
    else:
        raise ValueError('Activation norm layer %s '
                         'is not recognized' % norm_type)
    return norm_layer

def get_nonlinearity_layer(nonlinearity_type, inplace):
    r"""Return a nonlinearity layer.

    Args:
        nonlinearity_type (str):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace (bool): If ``True``, set ``inplace=True`` when initializing
            the nonlinearity layer.
    """
    if nonlinearity_type == 'relu':
        nonlinearity = nn.ReLU(inplace=inplace)
    elif nonlinearity_type == 'leakyrelu':
        nonlinearity = nn.LeakyReLU(0.2, inplace=inplace)
    elif nonlinearity_type == 'prelu':
        nonlinearity = nn.PReLU()
    elif nonlinearity_type == 'tanh':
        nonlinearity = nn.Tanh()
    elif nonlinearity_type == 'sigmoid':
        nonlinearity = nn.Sigmoid()
    elif nonlinearity_type.startswith('softmax'):
        dim = nonlinearity_type.split(',')[1] if ',' in nonlinearity_type else 1
        nonlinearity = nn.Softmax(dim=int(dim))
    elif nonlinearity_type == 'none' or nonlinearity_type == '':
        nonlinearity = None
    else:
        raise ValueError('Nonlinearity %s is not recognized' %
                         nonlinearity_type)
    return nonlinearity


class WeightDemodulation(nn.Module):
    r"""Weight demodulation in
    "Analyzing and Improving the Image Quality of StyleGAN", Karras et al.

    Args:
        conv (torch.nn.Modules): Convolutional layer.
        cond_dims (int): The number of channels in the conditional input.
        eps (float, optional, default=1e-8): a value added to the
            denominator for numerical stability.
        adaptive_bias (bool, optional, default=False): If ``True``, adaptively
            predicts bias from the conditional input.
        demod (bool, optional, default=False): If ``True``, performs
            weight demodulation.
    """

    def __init__(self, conv, cond_dims, eps=1e-8,
                 adaptive_bias=False, demod=True):
        super().__init__()
        self.conv = conv
        self.adaptive_bias = adaptive_bias
        if adaptive_bias:
            self.conv.register_parameter('bias', None)
            self.fc_beta = LinearBlock(cond_dims, self.conv.out_channels)
        self.fc_gamma = LinearBlock(cond_dims, self.conv.in_channels)
        self.eps = eps
        self.demod = demod
        self.conditional = True

    def forward(self, x, y):
        r"""Weight demodulation forward"""
        b, c, h, w = x.size()
        self.conv.groups = b
        gamma = self.fc_gamma(y)
        gamma = gamma[:, None, :, None, None]
        weight = self.conv.weight[None, :, :, :, :] * (gamma + 1)

        if self.demod:
            d = torch.rsqrt(
                (weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * d

        x = x.reshape(1, -1, h, w)
        _, _, *ws = weight.shape
        weight = weight.reshape(b * self.conv.out_channels, *ws)
        x = self.conv.conv2d_forward(x, weight)

        x = x.reshape(-1, self.conv.out_channels, h, w)
        if self.adaptive_bias:
            x += self.fc_beta(y)[:, :, None, None]
        return x

def weight_demod(conv, cond_dims=256, eps=1e-8, demod=True):
    r"""Weight demodulation."""
    return WeightDemodulation(conv, cond_dims, eps, demod)

def get_weight_norm_layer(norm_type, **norm_params):
    r"""Return weight normalization.

    Args:
        norm_type (str):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        norm_params: Arbitrary keyword arguments that will be used to
            initialize the weight normalization.
    """
    if norm_type == 'none' or norm_type == '':  # no normalization
        return lambda x: x
    elif norm_type == 'spectral':  # spectral normalization
        return functools.partial(spectral_norm, **norm_params)
    elif norm_type == 'weight':  # weight normalization
        return functools.partial(weight_norm, **norm_params)
    elif norm_type == 'weight_demod':  # weight demodulation
        return functools.partial(weight_demod, **norm_params)
    else:
        raise ValueError(
            'Weight norm layer %s is not recognized' % norm_type)

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
