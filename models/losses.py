import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # NOQA

import torch
import torch.nn as nn

# Based on https://github.com/knazeri/edge-connect/blob/master/src/loss.py
class GANLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, device, gan_type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge | l1
        """
        super(GANLoss, self).__init__()

        self.gan_type = gan_type
        self.register_buffer('real_label', torch.tensor(target_real_label).to(device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label).to(device))

        if gan_type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif gan_type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif gan_type == 'hinge':
            self.criterion = nn.ReLU()

        elif gan_type == 'l1':
            self.criterion = nn.L1Loss()

        elif self.gan_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()

        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def __call__(self, outputs, is_real, is_disc=None):
        """
        Args:
            outputs (Tensor): The input for the loss module, i.e., the network
                prediction.
            is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        if self.gan_type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss
