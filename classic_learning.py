import argparse
import pandas as pd
import tensorflow as tf
import dataset_ops
import numpy as np
import datetime
import cuda
from pathlib import Path
from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import train_test_split

try:
    import tqdm
except ImportError:
    tqdm = None

session_start = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('-w', nargs='+', type=int)
parser.add_argument('-o', '--out', default=f'classic_learning_{session_start}.csv')
parser.add_argument('models', nargs='+')

args = parser.parse_args()
print(args)

pd.set_option('display.max_columns', None)  # show all columns
cuda.initialize()

# dataset_manager = dataset_ops.MicroPilotTestsManager(dataset_dir=Path('./h5'), runs_filename='runs.hdf')
dataset_manager = dataset_ops.PaparazziTestManager(dataset_dir=Path('pprz_h5'), runs_filename='pprz_runs.hdf')
all_runs = dataset_manager.get_all_available_tests()


# selected_runs = all_runs.loc[(all_runs['Test Length'] > 200) & (all_runs['Test Length'] < 20000)]
# selected_runs = selected_runs.iloc[:40]
selected_runs = all_runs.sample(frac=1, axis=1, random_state=55)

inputs = ('SpeedFts', 'Pitch', 'Roll', 'Yaw', 'current_altitude', )
outputs= ('elev', 'ai', 'rdr', 'throttle', 'Flaps')

# max_length = selected_runs['Test Length'].max()
# max_length = 18000

# dataset_manager.preload_data(selected_runs, max_length=max_length, features=inputs + outputs)
# tfdataset = dataset_ops.TensorflowDataset(dataset_manager)
# dataset = tfdataset.get_dataset(selected_runs, batch_size=25, features=inputs+outputs, max_length=max_length)

dataset = dataset_manager.preload_data(selected_runs, features=inputs+outputs)
N_s = dataset_manager.count_states()
train, test = train_test_split(dataset, test_size=0.2, random_state=44)

# %%

# results_df = pd.DataFrame(columns=['Data Set', 'Name', 'regularization', 'd', 'w', 'Precision', 'Recall'])
# results_df = results_df.set_index(['Data Set', 'Name', 'w'])
# results_df = pd.read_csv('tree_results_20.csv', index_col=0).append(pd.read_csv('results_3_5_10_15.csv', index_col=0))

file_name = Path(args.out)

if file_name.exists():
    results_df = pd.read_csv(file_name, index_col=0)
else:
    results_df = pd.DataFrame(columns=['Data Set', 'Name', 'regularization', 'w', 'Precision', 'Recall'])


# %%

def class_precision_recall(y_true, y_pred):
    # _, classes = y_pred.shape

    # y_true = tf.math.argmax(y_true, axis=-1)  # batch x L
    # y_pred = tf.math.argmax(y_pred, axis=-1)

    y_true = tf.constant(y_true)
    y_pred = tf.constant(y_pred)

    classes = N_s

    y_pred.shape.assert_is_compatible_with(y_true.shape)
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)

    recall_scores, precision_scores = [], []
    for C in range(classes):
        C = tf.cast(C, 'int64')
        trueC = tf.equal(y_true, C)
        declaredC = tf.equal(y_pred, C)
        correctlyC = tf.logical_and(declaredC, trueC)

        trueC = tf.cast(tf.math.count_nonzero(trueC), 'float32')
        declaredC = tf.cast(tf.math.count_nonzero(declaredC), 'float32')
        correctlyC = tf.cast(tf.math.count_nonzero(correctlyC), 'float32')

        if declaredC > 0:
            precision_score = tf.math.divide_no_nan(correctlyC, declaredC)
            precision_scores.append(precision_score)
        if trueC > 0:
            recall_score = tf.math.divide_no_nan(correctlyC, trueC)
            recall_scores.append(recall_score)

    P = tf.reduce_mean(tf.stack(precision_scores))
    R = tf.reduce_mean(tf.stack(recall_scores))

    return P, R


def iterate_window(dataframe, w):
    last = dataframe.shape[0] - w + 1
    for index in range(last):
        yield dataframe[index:index + w]


def convert_to_data_points(w):
    def _generator(data):
        signals = data[1].to_numpy()
        states = data[2].to_numpy()
        signals_iter = iterate_window(signals, w)
        # previous_iter = iterate_window(states, w)
        next_state_iter = states[w:]

        # for signals, previous, next_state in zip(signals_iter, previous_iter, next_state_iter):
        for signals, next_state in zip(signals_iter, next_state_iter):
            # X = np.concatenate((signals.flatten(), one_hotter[previous].flatten()))
            X = signals.flatten()
            # y = one_hotter[next_state]
            y = next_state
            yield X, y

    return _generator


def create_allX_allY(dataset, w):
    converter = convert_to_data_points(w)
    allX, allY = [], []
    for test in tqdm.tqdm(dataset):
        lX, ly = [], []
        for X, y in converter(test):
            lX.append(X)
            ly.append(y)
        allX.append(np.stack(lX))
        allY.append(np.stack(ly))
        del lX, ly

    # return allX, allY
    return np.concatenate(allX), np.concatenate(allY)


def evaluate(model, *, name, w, regularization):
    train_p, train_r = class_precision_recall(y_train, model.predict(X_train))
    test_p, test_r = class_precision_recall(y_test, model.predict(X_test))
    df = pd.DataFrame({
        'Data Set': ['Train', 'Test'],
        'Precision': [float(train_p), float(test_p)],
        'Recall': [float(train_r), float(test_r)],
    })
    df['Name'] = name
    df['w'] = w
    df['regularization'] = regularization
    # df = df.set_index(['Data Set', 'Name', 'w'])

    return df


# %%

for w in args.w:
    print('Window size', w)
    X_train, y_train = create_allX_allY(train, w)
    X_test, y_test = create_allX_allY(test, w)

    # .set_index(['Data Set', 'Name', 'w'])
    if 'ridge' in args.models:
        print('Training Ridge')
        model_ridge = linear_model.RidgeClassifierCV(alphas=np.logspace(-6, 6, 13))
        model_ridge.fit(X_train, y_train)
        results = evaluate(model_ridge, name='ridge', w=w, regularization=None)

        if file_name.exists():
            results_df = pd.read_csv(file_name, index_col=0)
        results_df = results_df.append(results)
        print(results)
        try:
            results_df.to_csv(file_name)
        except:
            print('Failed to save')

    if 'tree1' in args.models or 'trees' in args.models:
        print('Training Tree')
        model_tree = tree.DecisionTreeClassifier(max_features=None)
        model_tree.fit(X_train, y_train)
        results = evaluate(model_tree, name='tree', regularization=None, w=w)

        if file_name.exists():
            results_df = pd.read_csv(file_name, index_col=0)
        results_df = results_df.append(results)
        print(results)
        print(model_tree.get_depth(), model_tree.get_n_leaves())
        try:
            results_df.to_csv(file_name)
        except:
            print('Failed to save')

    if 'tree2' in args.models or 'trees' in args.models:
        print('Training Tree (sqrt regularization)')
        model_tree = tree.DecisionTreeClassifier(max_features='sqrt')
        model_tree.fit(X_train, y_train)
        results = evaluate(model_tree, name='tree', regularization='sqrt', w=w)

        if file_name.exists():
            results_df = pd.read_csv(file_name, index_col=0)
        results_df = results_df.append(results)
        print(results)
        print(model_tree.get_depth(), model_tree.get_n_leaves())
        try:
            results_df.to_csv(file_name)
        except:
            print('Failed to save')

    if 'tree3' in args.models or 'trees' in args.models:
        print('Training Tree (log2 regularization)')
        model_tree = tree.DecisionTreeClassifier(max_features='log2')
        model_tree.fit(X_train, y_train)
        results = evaluate(model_tree, name='tree', regularization='log2', w=w)

        if file_name.exists():
            results_df = pd.read_csv(file_name, index_col=0)
        results_df = results_df.append(results)
        print(results)
        print(model_tree.get_depth(), model_tree.get_n_leaves())
        try:
            results_df.to_csv(file_name)
        except:
            print('Failed to save')

# results_df.to_csv('tree_results_15.csv')

# del X_train, X_test, y_train, y_test
exit(0)

























results_df.loc[(results_df['Data Set'] == 'Test') & (results_df['w'] == w)]

# %%

# results_df.loc['Test',results_df.reset_index()['Name'].str.contains('log2'), 5]
# results_df.loc[(results_df['Data Set'] == 'Test') & (results_df['w'] == w) & (results_df['regularization'] == 'log2')] \
#     .sort_values(['d', 'regularization'])
results_df.loc[(results_df['Data Set'] == 'Test') & (results_df['w'] == w)].sort_values(['regularization'])
# results_df

# %%

results_df['Precision'] *= 100
results_df['Recall'] *= 100
results_df['F1'] = 2 * results_df['Precision'] * results_df['Recall'] / (results_df['Precision'] + results_df['Recall'])
test_results = results_df.loc[results_df['Data Set'] == 'Test'].reset_index(drop=True)

# %%

test_results

# %%

ridge_results = test_results.loc[test_results['Name'] == 'ridge']
ridge_results

# %%

tree_results = test_results.loc[test_results['Name'] == 'tree']

# %%

tree_results.loc[tree_results['w'] == 3]

# %%

tree_results.loc[tree_results['w'] == 5]

# %%

tree_results.loc[tree_results['w'] == 10]

# %%

tree_results.loc[tree_results['w'] == 15]

# %%

tree_results.loc[tree_results['w'] == 20]

# %%

tree_results.loc[tree_results.groupby(['w'])['F1'].transform(max) == tree_results['F1']].sort_values('w')

# %%

ridge_results.loc[ridge_results.groupby(['w'])['F1'].transform(max) == ridge_results['F1']].sort_values('w')

# %%

results_df.shape


