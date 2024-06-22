# Adding Numbers with a Transformer

### Interactive demo: [https://siliconsloth.com/posts/adding-numbers-with-a-transformer/](https://siliconsloth.com/posts/adding-numbers-with-a-transformer/)

This project trains a transformer model to perform symbolic arithmetic on sequences of digits.
Given a sequence consisting of two numbers expressed as lists of digits 0-9, the model generates
the sum of the two numbers, as a list of digits. This demonstrates how transformers can learn
to perform simple algorithms, which will hopefully be scaled up to solve complex problems in the future.

## Usage

Run `train.py` to train the model. The script will stop running automatically once the model accuracy is high enough.

Run `test.py` to try out the model on input numbers of your choosing.

You can find an interactive demo of the model at [siliconsloth.com](https://siliconsloth.com/posts/adding-numbers-with-a-transformer/).

## Requirements

Requires PyTorch and TensorBoard to be installed.