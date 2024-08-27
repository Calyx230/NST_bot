# NST bot

This is a final project for the first part of [Deep Learning school](https://dls.samcs.ru/).

This project contains a telegram bot that performs style transfer via Gatys algorithm. The bot uses pretrained VGG-19 with default weights from PyTorch library. It also impelements Laplacian loss from [Laplacian-Steered Neural Style Transfer, Li et al.](https://arxiv.org/abs/1707.01253).
Bot's name is style_transfer_bot (@pointkittybot).

In `begin.py` the bot and the dispatcher are created, `handlers.py` contain message processing, and NST.py contains a neural network.

