# Transformer From Scratch

The goal of this project is to implement a transformer model from scratch using PyTorch and train it on some real-world datasets.

## Implementation and Training

### Transformer Models

The transformer model was introduced in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. The transformer model is based on the self-attention mechanism, which is a mechanism that allows the model to focus on different parts of the input sequence when predicting the output sequence. This repository contains an implementation of the transformer model using PyTorch. Two types of transformer models are implemented:

1. **Encoder-Decoder Transformer:** The standard transformer model introduced in the paper. This transformer model consists of an encoder and a decoder. The encoder processes the input sequence and produces a sequence of hidden states, which are then used by the decoder to generate the output sequence. The model is used along with a english to french translation dataset to train the model. Have a look at the [notebook](https://github.com/Hari31416/transformer_from_scratch/blob/main/notebooks/transformer_for_translation.ipynb) for more details. The trained model is saved in the `notebooks` directory.
2. **Decoder-Only Transformer:** A simplified version of the transformer model that only consists of a decoder. Most of the current LLMs are based on this architecture. We have trained the model on lyrics of Taylor Swift songs. Have a look at the [notebook](https://github.com/Hari31416/transformer_from_scratch/blob/main/notebooks/transformer_train_generation.ipynb) for more details. Again, the trained model is saved in the `notebooks` directory.

The source code for the transformer models is implemented in the `src/transformer.py` file. The implementation is heavily inspired by the [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).

### Utility Functions and Classes for Training

Apart from the two types of transformer model, a number of utility functions and classes are also implemented to facilitate the training and evaluation of the models. This includes classes for creating the dataset (for translation and generation), handling source and target sequences, training the model, and generating output sequences. All the utility functions and classes are implemented in the `src/train_utils.py` file.

### Data Used

Both the data used and their tokenizers are saved in the `data` directory. The data sources are:

1. **English to French Translation Dataset:** The dataset is downloaded from the kaggle dataset [Language Translation (English-French)](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench/data). No preprocessing is done to this dataset except from renaming the columns.
2. **Lyrics of Taylor Swift Songs:** I have used a subset of the [Genius Song Lyrics](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information) from Kaggle. Some preprocessing is done to this dataset. Look at the [notebook](https://github.com/Hari31416/transformer_from_scratch/blob/main/notebooks/lyrics.ipynb) for details.
