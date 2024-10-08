import tensorflow as tf

# Check if TensorFlow can access a GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"Number of GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU Name: {gpu.name}")
else:
    print("No GPUs available.")
