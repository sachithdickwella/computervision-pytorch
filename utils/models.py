#!/usr/bin/env python3
# -*- encode: utf-8 -*-

import torch
import torch.nn as nn

import time
import numpy as np
import skimage.color as color


def lab2rgb(image, dim=None):
    """
    :param image is a single Lab image the shape of (X, y, 3)
    :param dim to define the color-channel dimension.
    """
    image = np.transpose(image, axes=(1, 2, 0))

    if dim is not None:
        z = np.zeros_like(image)
        if dim != 0:
            z[:, :, 0] = 80  # Increase the brightness to see other color channels(a & b).

        z[:, :, dim] = image[:, :, dim]
        return color.lab2rgb(z)
    else:
        return color.lab2rgb(image)


def train(model, epochs, x_loader, y_loader, device='cpu'):
    torch.cuda.empty_cache()

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start = time.time()
    losses = []

    for i in range(epochs):
        i += 1

        for batch, (X, y) in enumerate(zip(x_loader, y_loader)):
            batch += 1

            X = X[0].to(device)
            y = y[0].to(device)

            y_pred = model(X)
            loss = criterion(y_pred, y)

            losses.append(loss)

            if batch == 1 or batch % 100 == 0:
                print(f'Epoch: {i}/{epochs}, Batch: {batch}/{len(x_loader)} => Loss: {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    duration = time.time() - start
    print(f'Duration to execute: {duration / 60:0.4f} minutes')

    return losses


class ColorNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=3, n_layers=(32, 128, 64, 8)):
        super(ColorNet, self).__init__()

        layers = []
        prev = in_channels

        for n in n_layers:
            layers.append(nn.Conv2d(in_channels=prev, out_channels=n, kernel_size=4, stride=2, padding=1)),
            layers.append(nn.BatchNorm2d(n)),
            layers.append(nn.ReLU())
            layers.append(nn.Upsample(scale_factor=2.))
            prev = n

        layers.append(nn.Conv2d(in_channels=prev, out_channels=out_channels, kernel_size=3, stride=1, padding=1))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
