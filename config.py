# Percentage of dataset to use for training; rest is used for validation
percentage_training = 0.9
# Input parameters
batch_size = 128
img_rows, img_cols = 256, 256
kernel_size = 3
# Number of colors in color space
num_colors = 313
# directories
data_dir = 'data/'
imgs_dir = 'generated_dataset/generated_dataset'
checkpoint_models_path = 'models/'
# Parameters for random dataset generation from imagenet
train_set_dim = 1024  # mb

# Number of neighbours for smoothing
nb_neighbors = 5

# Training parameters
fmt = '.jpeg'
# he normal = truncated normal distribution centered on 0
layer_init = 'he_normal'
save_period = 10
patience = 50
epochs = 1000
learning_rate = 0.0001
T = 0.1
