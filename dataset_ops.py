import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from tqdm import notebook as tqdm
except ImportError:
    tqdm = None


class TestsManager(object):
    state_cols = []

    def __init__(self, dataset_dir='./h5', runs_filename='runs.hdf'):
        if not isinstance(dataset_dir, Path):
            dataset_dir = Path(str(dataset_dir))
        if not isinstance(runs_filename, Path):
            runs_filename = Path(str(runs_filename))

        self.dataset_dir = dataset_dir
        self.runs_filename = runs_filename
        self.test_data_cache = dict()
        self.unique_states = defaultdict(self.count_states)
        self._signal_targets_prefetched = None

    def count_states(self):
        return len(self.unique_states)

    def _state_ids(self, df: pd.DataFrame):
        df = df[self.state_cols]
        change_locations = (df != df.shift()).apply(lambda x: np.logical_or.reduce(x.to_numpy()), axis=1)
        #     change_locations = np.squeeze(np.where(change_locations))
        #     change_locations = np.delete(change_locations, np.where(change_locations==0))
        #     self.unique_states = defaultdict(lambda *_: len(self.unique_states))
        return df.loc[change_locations].apply(lambda row: self.unique_states[tuple(x[1] for x in sorted(row.items()))],
                                             axis=1)

    #     return changeLocations

    def _generate_signals_targets(self, selected_runs: pd.DataFrame, *, features: List[str],
                                  max_length: Optional[int] = None):
        runs_iter = selected_runs.iterrows()
        if tqdm:
            runs_iter = tqdm.tqdm(runs_iter, total=selected_runs.shape[0])

        for tid, run in runs_iter:
            tdata, _ = self.read_test_data(tid, run)
            signals = tdata[[*features]]
            targets = tdata['state_id']

            if max_length is not None and signals.shape[0] > max_length:
                signals = signals.iloc[:max_length]
                targets = targets.iloc[:max_length]

            yield tid, signals, targets

    def preload_data(self, selected_runs: pd.DataFrame, *, features: List[str], max_length: Optional[int] = None):
        if self._signal_targets_prefetched is None:
            print("Loading data", end='...')
            self._signal_targets_prefetched = [
                *self._generate_signals_targets(selected_runs, features=features, max_length=max_length)]
            print("done")

        return self._signal_targets_prefetched

    def iterate(self):
        assert self._signal_targets_prefetched is not None, "Fetch the data first"

        return self._signal_targets_prefetched

    def get_all_available_tests(self):
        raise NotImplementedError()

    def read_test_data(self, tid, run):
        raise NotImplementedError()


class MicroPilotTestsManager(TestsManager):
    state_cols = ['Laileron', 'Lelevator', 'Lflap', 'Lrudder', 'Lthrottle']

    def __init__(self, dataset_dir='./h5', runs_filename='runs.hdf'):
        super(MicroPilotTestsManager, self).__init__(dataset_dir, runs_filename)
        self.commands = pd.read_csv('COMMANDS.csv', header=None, index_col=1, names=['name'])
        self._signal_targets_prefetched = None

    def get_all_available_tests(self):
        if os.path.exists(self.runs_filename):
            return pd.read_hdf(self.runs_filename, 'runs')

        all_runs = []
        for name in os.listdir(self.dataset_dir):
            if name[-3:] != '.h5':
                continue

            testid, planeid = name[:-3].split('_', 2)
            tdata, states = self.read_test_data({'Test ID': testid, 'PlaneId': planeid})
            n, m = tdata.shape[0], tdata.index.max()
            if n != m:
                print(f'panic for test {testid}, n={n}, m={m}, missing={set(range(1, m)) - set(tdata.index)}')

            all_runs.append((testid, planeid, n))

        all_runs = pd.DataFrame(all_runs, columns=('Test ID', 'PlaneId', 'Test Length'))
        all_runs.to_hdf(self.runs_filename, 'runs')

        return all_runs

    def read_test_data(self, _, run):
        testid, planeid = run['Test ID'], run['PlaneId']
        if (testid, planeid) not in self.test_data_cache:
            tdata: pd.DataFrame = pd.read_hdf(f'{self.dataset_dir}/{testid}_{planeid}.h5',
                                              key=f'TestData_{testid}_{planeid}')
            tdata['state_id'] = np.nan
            maxT = tdata.index.max()

            states = self._state_ids(tdata)
            nice_to_remove = {t1 for t1, t2 in zip(states.index, states.index[1:]) if t2 - t1 < 5}
            states.drop(nice_to_remove, axis=0, inplace=True)
            tdata['state_id'] = states.iloc[0]
            for (t1, s1), t2 in zip(states.items(), states.index[1:]):
                tdata.loc[t1:t2, 'state_id'] = s1
            t1, t2, s1 = states.iloc[[-1]].index[0], maxT, states.iloc[-1]
            tdata.loc[t1:t2, 'state_id'] = s1

            self.test_data_cache[(testid, planeid)] = tdata, states

        return self.test_data_cache[(testid, planeid)]

    def command_names(self, df: pd.DataFrame):
        return df['currentCommandId'].apply(lambda cid: self.commands.loc[cid])


class PaparazziTestManager(TestsManager):
    state_cols = ['ap_gaz', 'ap_lateral', 'ap_horizontal', 'v_ctl_auto_throttle_submode', 'v_ctl_climb_mode',
                  'h_ctl_pitch_mode', 'nav_mode']

    def __init__(self, dataset_dir='./pprz_h5', runs_filename='pprz_runs.hdf'):
        super(PaparazziTestManager, self).__init__(dataset_dir, runs_filename)

    def get_all_available_tests(self):
        run_file = self.runs_filename
        if run_file.exists():
            return pd.read_hdf(run_file, 'runs')

        test_number = re.compile(r'test_(\d+)\.h5')

        test_lengths = []
        directory = Path(__file__).parent / 'pprz_h5'
        for file in directory.iterdir():
            if not file.is_file() or file.suffix != '.h5':
                continue
            idx = int(re.findall(test_number, file.name)[0])
            try:
                df = pd.read_hdf(file)
                test_lengths.append((idx, df.shape[0]))
                del df
            except:
                print(file)  # TODO: Bad practice in catching the error and in hadnling it. Should be fixed

        test_lengths_df = (pd.DataFrame(test_lengths, dtype='int32', columns=['idx', 'Test Length'])
                           .set_index('idx', drop=True)
                           .sort_index())
        test_lengths_df.to_hdf(run_file, key='runs')

        return test_lengths_df

    def read_test_data(self, run_id, _):
        if run_id not in self.test_data_cache:
            tdata: pd.DataFrame = pd.read_hdf(self.dataset_dir / f'test_{run_id}.h5')  # noqa: it's going to be a df
            tdata['state_id'] = np.nan
            max_t = tdata.index.max()

            states = self._state_ids(tdata)
            nice_to_remove = {t1 for t1, t2 in zip(states.index, states.index[1:]) if t2 - t1 < 5}
            states.drop(nice_to_remove, axis=0, inplace=True)
            tdata['state_id'] = states.iloc[0]
            for (t1, s1), t2 in zip(states.items(), states.index[1:]):
                tdata.loc[t1:t2, 'state_id'] = s1
            t1, t2, s1 = states.iloc[[-1]].index[0], max_t, states.iloc[-1]
            tdata.loc[t1:t2, 'state_id'] = s1

            self.test_data_cache[run_id] = tdata, states

        return self.test_data_cache[run_id]


class TensorflowDataset:
    def __init__(self, dataset_manager: TestsManager):
        self.dataset_manager = dataset_manager

    def _create_padded_generator(self, selected_runs: pd.DataFrame, *, features: List[str], max_length: int):
        self.dataset_manager.preload_data(selected_runs, max_length=max_length, features=features)
        N_s = self.dataset_manager.count_states()

        def _gen():
            for tid, signals, targets in self.dataset_manager.iterate():
                s = tf.convert_to_tensor(signals.to_numpy(), dtype='float32')
                t = tf.one_hot(targets.to_numpy(), depth=N_s)

                length = s.shape[0]

                mask = tf.fill((length, 1), True)

                pad_size = max_length - length
                paddings = tf.constant([[0, pad_size], [0, 0]])
                s = tf.pad(s, paddings, 'CONSTANT')
                t = tf.pad(t, paddings, 'CONSTANT')
                mask = tf.pad(mask, paddings, 'CONSTANT', constant_values=False)
                yield {'signals': s, 'mask': mask}, t

        return _gen

    def get_dataset(self, selected_runs: pd.DataFrame, *, features: List[str], max_length: int) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(
            self._create_padded_generator(selected_runs, features=features, max_length=max_length),
            output_types=({'signals': tf.float32, 'mask': tf.float32}, tf.float32),
            output_shapes=({
                               'signals': tf.TensorShape([max_length, len(features)]),
                               'mask': tf.TensorShape([max_length, 1])
                           },
                           tf.TensorShape([max_length, self.dataset_manager.count_states()])),
        )

        return ds


def split_dataset(ds, split_proportion: Tuple[int, int, int]):
    def get_second(_, x):
        return x

    limits = [0]
    for p in split_proportion:
        limits.append(limits[-1] + p)

    def make_sieve(idx):
        lower, upper, total = limits[idx], limits[idx + 1], limits[-1]

        def _sieve(index, _):
            r = index % total
            return (lower <= r) and (r < upper)
        return tf.function(_sieve)

    return tuple(
        ds.enumerate().filter(make_sieve(idx)).map(get_second)
        for idx in range(len(split_proportion))
    )


def load_and_split(dataset_manager, selected_runs, features, split_ratio, batch_size, max_length=None):
    if not max_length:
        max_length = selected_runs['Test Length'].max()

    tfdataset = TensorflowDataset(dataset_manager)
    ds = tfdataset.get_dataset(selected_runs, features=features, max_length=max_length)
    train_dataset, test_dataset, validation_dataset = split_dataset(ds, split_proportion=split_ratio)
    train_dataset, test_dataset, validation_dataset = (
        dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            .batch(batch_size)
            .shuffle(buffer_size=15)
        for dataset in (train_dataset, test_dataset, validation_dataset)
    )

    return ds, train_dataset, test_dataset, validation_dataset

