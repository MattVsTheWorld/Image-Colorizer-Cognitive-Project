# Percentage of dataset to use for training; rest is used for validation
percentage_training = 0.99
# Input parameters
batch_size = 64
img_rows, img_cols = 256, 256
kernel_size = 3
# Number of colors in color space
num_colors = 313
# directories
data_dir = 'data/'
imgs_dir = 'dataset/'
checkpoint_models_path = 'models/'
# Parameters for random dataset generation from imagenet
train_set_dim = 1024  # mb

# Number of neighbours for smoothing
nb_neighbors = 5

# Training parameters
fmt = '.jpeg'
# he normal = truncated normal distribution centered on 0
layer_init = 'he_normal'
# Save every # epochs
save_period = 1
# Stop training if validation loss does not improve for # epochs
patience = 50
# The number of epochs is unimportant; training stops after the patience period
epochs = 1000
learning_rate = 3.16e-5
T = 0.38
