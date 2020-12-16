"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

import torch
from torch import nn
from types import SimpleNamespace
from functools import partial
from torch import autograd

from funit_from_imaginaire import *

class GPPatchMcResDis(nn.Module):
    def __init__(self, hp):
        super(GPPatchMcResDis, self).__init__()
        assert hp['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = hp['n_res_blks'] // 2
        padding_mode = 'reflect'
        weight_norm_type = 'spectral'
        image_channels = 4
        num_filters = hp['nf']
        num_layers = self.n_layers
        max_num_filters = 1024
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

        self.embedder = nn.Embedding(hp['num_classes'], num_filters)

    def forward(self, images, labels):
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

    def calc_dis_fake_loss(self, input_fake, input_label):
        resp_fake, gan_feat = self.forward(input_fake, input_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real, input_label):
        resp_real, gan_feat = self.forward(input_real, input_label)
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).cuda()
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        loss = -resp_fake.mean()
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()/batch_size
        return reg

class FewShotGen(nn.Module):
    def __init__(self, hp):
        super(FewShotGen, self).__init__()
        nf = hp['nf']
        nf_mlp = hp['nf_mlp']
        down_class = hp['n_downs_class']
        down_content = hp['n_downs_content']
        n_mlp_blks = hp['n_mlp_blks']
        n_res_blks = hp['n_res_blks']
        latent_dim = hp['latent_dim']

        self.enc_class_model = StyleEncoder(num_downsamples = down_class,
                                            image_channels = 4,
                                            num_filters = nf,
                                            style_channels = latent_dim,
                                            padding_mode = 'reflect',
                                            activation_norm_type = 'none',
                                            weight_norm_type = '',
                                            nonlinearity = 'relu')

        self.enc_content = ContentEncoder(num_downsamples = down_content,
                                              num_res_blocks = n_res_blks,
                                              image_channels = 4,
                                              num_filters = nf,
                                              padding_mode = 'reflect',
                                              activation_norm_type = 'instance',
                                              weight_norm_type = '',
                                              nonlinearity = 'relu')

        self.dec = Decoder(self.enc_content.output_dim,
                               nf_mlp,
                               4,
                               down_content,
                               'reflect',
                               '',
                               'relu')

        usb_dims = 1024
        self.usb = torch.nn.Parameter(torch.randn(1, usb_dims))

        self.mlp = MLP(latent_dim,
                       nf_mlp,
                       nf_mlp,
                       n_mlp_blks,
                       'none',
                       'relu')

        num_content_mlp_blocks = 2
        num_style_mlp_blocks = 2
        self.mlp_content = MLP(self.enc_content.output_dim,
                               latent_dim,
                               nf_mlp,
                               num_content_mlp_blocks,
                               'none',
                               'relu')

        self.mlp_style = MLP(latent_dim + usb_dims,
                             latent_dim,
                             nf_mlp,
                             num_style_mlp_blocks,
                             'none',
                             'relu')

    def forward(self, one_image, model_set):
        # reconstruct an image
        content, model_codes = self.encode(one_image, model_set)
        model_code = torch.mean(model_codes, dim=0).unsqueeze(0)
        images_trans = self.decode(content, model_code)
        return images_trans

    def encode(self, one_image, model_set):
        # extract content code from the input image
        content = self.enc_content(one_image)
        # extract model code from the images in the model set
        class_codes = self.enc_class_model(model_set)
        class_code = torch.mean(class_codes, dim=0).unsqueeze(0)
        return content, class_code

    def decode(self, content, style):
        content_style_code = content.mean(3).mean(2)
        content_style_code = self.mlp_content(content_style_code)
        batch_size = style.size(0)
        usb = self.usb.repeat(batch_size, 1)
        style = style.view(batch_size, -1)
        style_in = self.mlp_style(torch.cat([style, usb], 1))
        coco_style = style_in * content_style_code
        coco_style = self.mlp(coco_style)
        images = self.decoder(content, coco_style)
        return images

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

class MLP(nn.Module):
    r"""Improved FUNIT style decoder.

    Args:
        input_dim (int): Input dimension (style code dimension).
        output_dim (int): Output dimension (to be fed into the AdaIN
           layer).
        latent_dim (int): Latent dimension.
        num_layers (int): Number of layers in the MLP.
        activation_norm_type (str): Activation type.
        nonlinearity (str): Nonlinearity type.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 latent_dim,
                 num_layers,
                 activation_norm_type,
                 nonlinearity):
        super().__init__()
        model = []
        model += [LinearBlock(input_dim, latent_dim,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity)]
        # changed from num_layers - 2 to num_layers - 3.
        for i in range(num_layers - 3):
            model += [LinearBlock(latent_dim, latent_dim,
                                  activation_norm_type=activation_norm_type,
                                  nonlinearity=nonlinearity)]
        model += [LinearBlock(latent_dim, output_dim,
                              activation_norm_type=activation_norm_type,
                              nonlinearity=nonlinearity)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input tensor.
        """
        return self.model(x.view(x.size(0), -1))
