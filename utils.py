import matplotlib.pyplot as plt
import numpy as np


# Visualise autoencoder's representation
def display_reconstruction(model, x):
   # Compute corresponding representations
   original = np.reshape(x, (x.shape[0], 28, 28))
   reconstructed = np.reshape(model(x), (x.shape[0], 28, 28))

   # Plot the images
   n = len(x)  # how many images we will display
   plt.figure(figsize=(20, 4))
   
   for i in range(n):
      # Display original...
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(original[i])
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      
      # ... and its reconstruction
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(reconstructed[i])
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
   
   plt.show()
