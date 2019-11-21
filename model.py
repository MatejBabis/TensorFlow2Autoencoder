import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
   """
   Encoder component that learns structured data representation
   """
   def __init__(self, hidden_layer_dim, bottleneck_layer_dim):
      super(Encoder, self).__init__()
      
      self.hidden_layer = tf.keras.layers.Dense(
         units=hidden_layer_dim,
         activation=tf.nn.relu,
      )
      
      self.bottleneck_layer = tf.keras.layers.Dense(
         units=bottleneck_layer_dim,
         activation=tf.nn.sigmoid
      )
   
   def call(self, x):
      activation = self.hidden_layer(x)
      return self.bottleneck_layer(activation)


class Decoder(tf.keras.layers.Layer):
   def __init__(self, hidden_layer_dim, original_dim):
      super(Decoder, self).__init__()
      self.hidden_layer = tf.keras.layers.Dense(
         units=hidden_layer_dim,
         activation=tf.nn.relu,
      )
      self.output_layer = tf.keras.layers.Dense(
         units=original_dim,
         activation=tf.nn.sigmoid
      )
   
   def call(self, x):
      activation = self.hidden_layer(x)
      return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
   """
   Custom model combining the Encoder and Decoder sub-models
   """
   def __init__(self, hidden_layer_dim, bottleneck_layer_dim, original_dim):
      super(Autoencoder, self).__init__()
      self.encoder = Encoder(
         hidden_layer_dim=hidden_layer_dim,
         bottleneck_layer_dim=bottleneck_layer_dim
      )
      self.decoder = Decoder(
         hidden_layer_dim=hidden_layer_dim,
         original_dim=original_dim
      )
   
   def call(self, input_features):
      encoded = self.encoder(input_features)
      reconstructed = self.decoder(encoded)
      return reconstructed
