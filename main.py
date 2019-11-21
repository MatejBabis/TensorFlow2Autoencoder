import numpy as np
import tensorflow as tf
from model import Autoencoder, Encoder, Decoder
from utils import display_reconstruction

# Fix randomness
np.random.seed(1)
tf.random.set_seed(1)

# Hyperparams
BATCH_SIZE = 100
N_EPOCHS = 15
LEARNING_RATE = 1e-2
HIDDEN_DIM = 64
BOTTLENECK_DIM = 8

# Load the dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# Flatten the images to 1-d arrays
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# Transform it into tf Dataset objects
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices(x_test).shuffle(10000).batch(BATCH_SIZE)

# Network architecture with dimensionality
#  original_dim : hidden_dim : bottleneck_dim : hidden_dim : original_dim
autoencoder = Autoencoder(
   hidden_layer_dim=HIDDEN_DIM,
   bottleneck_layer_dim=BOTTLENECK_DIM,
   original_dim=x_train.shape[1]    # output dim is the same as input dim
)

# Evaluation metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

# Training optimizer
adam = tf.optimizers.Adam(learning_rate=LEARNING_RATE)


# Error metric for the auto-encoder representations
def loss_f(x, x_bar):
   return tf.losses.mean_squared_error(x, x_bar)

# Define the training process
@tf.function
def train_step(x):
   with tf.GradientTape() as tape:
      # Forward propagation of x
      x_reconstructed = autoencoder(x)
      loss = loss_f(x, x_reconstructed)
   # Compute the gradient w.r.t. tape context
   gradients = tape.gradient(loss, autoencoder.trainable_variables)
   adam.apply_gradients(zip(gradients, autoencoder.trainable_variables))
   # Store training loss for evaluation
   train_loss(loss)
   
# Validation process analogous to training
@tf.function
def test_step(x):
   x_reconstructed = autoencoder(x)
   loss = loss_f(x, x_reconstructed)
   test_loss(loss)
  

# Driver
for epoch in range(N_EPOCHS):
   for x_batch in train_ds:
      train_step(x_batch)
   for test_x_batch in test_ds:
      test_step(test_x_batch)
   # Result logging
   template = 'Epoch {}, Loss: {}, Test Loss: {}'
   print(template.format(epoch+1, train_loss.result(), test_loss.result()))
   
   # Reset the metrics for the next epoch
   train_loss.reset_states()
   test_loss.reset_states()
   
   # Visualise results on some randomly selected test cases
   rand_xs = x_test[np.random.randint(0, len(x_test), size=10)]
   display_reconstruction(autoencoder, rand_xs)
