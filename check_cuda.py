import tensorflow as tf
from tensorflow.python.client import device_lib

#is_available = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
#print('Available = ', is_available)
#print()
print(device_lib.list_local_devices())

print(tf.config.list_physical_devices())
