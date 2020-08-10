import tensorflow as tf
from tensorflow.python.client import device_lib


def initialize():
    GPUs = tf.config.list_physical_devices('GPU')
    if GPUs is None or len(GPUs) == 0:
        print("WARNING: No GPU, all there is is:")
        for device in tf.config.list_physical_devices():
            print(f'- {device}')
    else:
        for gpu in GPUs:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("Initialized", gpu)


if __name__ == '__main__':
    is_available = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    print('Available = ', is_available)
    print()
    print(device_lib.list_local_devices())

    print(tf.config.list_physical_devices())
