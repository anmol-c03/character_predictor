# Next character prediction using wavenet


# Table of Contents
1. [Project Overview](#project-overview)
3. [Model](#model)
4. [Results](#results)
7. [References](#references)

# Project Overview
    This project aims to predict new names by training a model on thousands of existing names. The WaveNet architecture, originally designed for generating raw audio waveforms, is used for character generation. WaveNet is a deep neural network that's fully probabilistic and autoregressive, meaning each output depends on all previous inputs. Although it was designed for audio data, in this project, it has been adapted to generate sequences of characters.A common problem with neural networks is that when inputs are combined into a single hidden layer, valuable information can be lost or not mapped correctly. To address this, the project forms pairs of tokens from the original input sequence and processes them through a feed-forward network. This approach helps retain more information during processing.The api.py file, which forms a core component of this project, is based on the WaveNet work of Andrej Karpathy.This project is primarily designed to enhance my learning and improve my familiarity with concepts related to language modeling 

# model

The model is built on the WaveNet architecture, where pairs of tokens are fed through a feed-forward network. This network comprises several layers: an embedding layer, a FlattenC layer, a linear layer, batch normalization, and a Tanh activation function.

Embedding Layer: This layer embeds tokens into a 24-dimensional space, allowing the model to represent each token with a specific vector.
FlattenC Layer: This is used to combine token pairs into a single representation.
Linear Layer: Similar to nn.Linear in PyTorch, it applies a linear transformation to the input data.
Batch Normalization: This layer normalizes the output of the linear layer, helping with training stability and convergence.
Tanh Activation: It applies the hyperbolic tangent function to add non-linearity to the network.
    ![wavenet_architecture](wavenet/wavenet.png)

#results
Training_loss:1.78427481
validation_loss:1.978255152
Names generated:
kayla.
slant.
carius.
kalani.
lexande.
kayron.
jalys.
rilinger.
phoenix.
emmauri.

#refrences
[!https://arxiv.org/pdf/1609.03499]
[!https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf]
