import argparse
from pathlib import Path

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd
import tensorflow as tf

import dataset_ops
from metrics import soft_dice_loss
from model_helper import MaskStealingLayer

try:
    from tqdm import notebook as tqdm
except ImportError:
    tqdm = None

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', choices={'training', 'testing', 'evaluation'}, default='training')
parser.add_argument('-c', '--cut', nargs=argparse.REMAINDER, type=int, default=[900])
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('i', type=int, metavar='skip', default=0)
parser.add_argument('model_weights')

args = parser.parse_args()
pd.set_option('display.max_columns', None)  # show all columns
if not args.no_cuda:
    import cuda
    cuda.initialize()

model_path = str((Path('models') / (args.model_weights + '.h5')).resolve())
model = tf.keras.models.load_model(model_path, custom_objects={
    'MaskStealingLayer': MaskStealingLayer,
    'soft_dice_loss': soft_dice_loss,
}, compile=False)


def plot_model_intermediate_layers(model, input_values, out_path, cut):
    from sklearn.preprocessing import minmax_scale

    if not isinstance(out_path, Path):
        out_path = Path(str(out_path))
    if not out_path.exists():
        out_path.mkdir(parents=True)

    functors = [tf.keras.backend.function([model.input], [layer.output]) for layer in model.layers if layer.name.startswith('conv_')]
    layer_outs = [func(input_values) for func in functors]

    for (disposition, layer), filter_index in zip(enumerate(layer_outs), [-4, -3, -5, -8]):
        plt.plot(minmax_scale(layer[0][0, cut, filter_index]) + disposition)
    plt.axis('off')
    plt.savefig(out_path / 'signal-out.png', bbox_inches='tight')
    plt.close()

    i = input_values[0][0]['signals']
    for disposition, filter_index in enumerate([1, 2, 5, 6]):
        plt.plot(minmax_scale(i[0, cut, filter_index]) + disposition)

    plt.axis('off')
    plt.savefig(out_path / 'signal.png', bbox_inches='tight')
    plt.close()


data_set_name = args.dataset
dataset_manager = dataset_ops.PaparazziTestManager(dataset_dir=Path('pprz_h5'), runs_filename='pprz_runs.hdf')
all_runs = dataset_manager.get_all_available_tests()

inputs = ('SpeedFts', 'Pitch', 'Roll', 'Yaw', 'current_altitude')
outputs= ('elev', 'ai', 'rdr', 'throttle', 'Flaps')

max_length = all_runs['Test Length'].max()
tfdataset = dataset_ops.TensorflowDataset(dataset_manager)
dataset = tfdataset.get_dataset(all_runs, batch_size=1, features=inputs+outputs, max_length=max_length)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

_filter = None
if data_set_name == 'training':
    _filter = lambda x, y: x % 10 > 3
elif data_set_name == 'testing':
    _filter = lambda x, y: x % 10 < 3
elif data_set_name == 'evaluation':
    _filter = lambda x, y: x % 10 == 3
else:
    raise ValueError('Should not happen')
dataset = dataset.enumerate().filter(_filter).map(lambda x, y: y)

plot_model_intermediate_layers(
    model=model,
    input_values=[*dataset.skip(args.i).take(1).as_numpy_iterator()],
    out_path='paper_data/ICSE',
    cut=slice(*args.cut)
)
