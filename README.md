# Learning Algorithms with a Transformer

### Interactive demo: [https://siliconsloth.com/posts/learning-algorithms-with-a-transformer/](https://siliconsloth.com/posts/learning-algorithms-with-a-transformer/)

This project trains a transformer model to run the bubble sort algorithm.
Given a list of digits 0-9, the model generates the intermediate steps of bubble sort until the list is sorted.
This demonstrates how transformers can learn to perform multi-step algorithms of varying length,
which will hopefully be scaled up to solve complex problems in the future.

## Usage

Run `train.py` to train the model. The script will save model checkpoints periodically
and can be stopped once the accuracy is high enough.

Run `test.py` to try out the model on input lists of your choosing.

You can find an interactive demo of the model at [siliconsloth.com](https://siliconsloth.com/posts/learning-algorithms-with-a-transformer/).

## Requirements

Requires PyTorch and TensorBoard to be installed.

## Data Generation

Algoformer generates lists of uniformly distributed length for training.
Some of the resulting bubble sort traces are very long,
such that the batch size needs to be varied to allow longer sequences to fit in GPU memory
while maintaining larger batches for shorter sequence lengths.
To this end, Algoformer sorts the generated training sequences into buckets based on length,
so that similar length sequences can be batched together using an appropriate batch size.

Another issue is that the bubble sort trace length can vary wildly based on the input list,
with particularly long traces being very under-represented in the underlying data generation distribution.
To make the length distribution in the training data more uniform, Algoformer caches a data batch for each
length bucket described above and randomly replaces samples in the batch as new ones are generated. This means
longer sequences that appear more rarely in the generation distribution can be saved and reused until new ones
are generated to replace them. The bucket batches are selected uniformly at random for each training step,
ensuring that longer sequences appear regularly in the training data.