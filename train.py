# global
import argparse
import pickle
import numpy as np
import sklearn.model_selection
import os.path
import shutil
import datetime
import tensorflow as tf
import keras
import sklearn
import functools
import subprocess
import importlib
import random
import pathlib
import h5py
from collections import namedtuple
# project local
import utils.sklearn
import utils.keras
import experimental.rf_vs_bmode.pmaps.piston.h5_data_provider as data_provider


def init_seeds():
    np.random.seed(42)
    random.seed(777)
    tf.set_random_seed(207)

no_test_splits = 1
test_size = .3
split_seed = 24

EPOCHS = 500

test_metrics = [
    ('mean_absolute_error', sklearn.metrics.mean_absolute_error, 'regression')
]

PointEvalResults = namedtuple("PointEvalResults", [
    "best_estimator",
    "history"
])


def create_fcn_layers(input_shape, units, hidden_activation, output_activation):
    layers = []
    if len(units) > 1:
        if input_shape is not None:
            first_layer = keras.layers.Dense(units[0], activation=None, input_shape=input_shape)
        else:
            first_layer = keras.layers.Dense(units[0], activation=None)
        layers.extend([
            first_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Activation(hidden_activation)
        ])
        for unit in units[1:-1]:
            layers.extend([
                keras.layers.Dense(unit, activation=None),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(hidden_activation)
            ])
        layers.append(
            keras.layers.Dense(units[-1], activation=output_activation)
        )
    else:
        if input_shape is not None:
            layers.append(
                keras.layers.Dense(units[-1], activation=None, input_shape=input_shape)
            )
        else:
            layers.append(
                keras.layers.Dense(units[-1], activation=None)
            )
    return layers

def create_conv_block(input_shape, number_of_kernels, kernel_size, pool_size, hidden_activation):
    conv_layer_parameters = {
        'filters' : number_of_kernels,
        'kernel_size' : kernel_size,
        'padding': "valid"
    }
    if input_shape is not None:
        conv_layer_parameters['input_shape'] = input_shape
    return [
        keras.layers.Conv1D(**conv_layer_parameters),
        keras.layers.BatchNormalization(),
        keras.layers.Activation(hidden_activation),
        keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_size)
    ]

ConvBlockDef = namedtuple("ConvBlockDef", [
    "number_of_kernels",
    "kernel_size",
    "pool_size"
])

def create_conv_blocks(input_shape, conv_blocks, hidden_activation):
    layers = []
    layers.extend(create_conv_block(
        input_shape,
        number_of_kernels=conv_blocks[0].number_of_kernels,
        kernel_size=conv_blocks[0].kernel_size,
        pool_size=conv_blocks[0].pool_size,
        hidden_activation='relu'
    ))
    for conv_block in conv_blocks[1:]:
        layers.extend(create_conv_block(
            input_shape=None,
            number_of_kernels=conv_block.number_of_kernels,
            kernel_size=conv_block.kernel_size,
            pool_size=conv_block.pool_size,
            hidden_activation='relu'
        ))

    return layers

def create_model(X, y, groups, path):
    """
    X_train, y_train - INDICES
    """

    # >>>>>>>>>>>>>>>>>>>>>> DATA:
    # Split to validation and training dataset.
    inner_cv = sklearn.model_selection.GroupShuffleSplit(
        n_splits=1,
        test_size=.2,
        random_state=42)

    train, val = next(inner_cv.split(X, groups=groups))
    X_train, y_train, groups_train = X[train], y[train], groups[train]
    X_val, y_val, groups_val = X[val], y[val], groups[val]

    assert not set(groups_train).intersection(set(groups_val))
    assert set(groups_train).union(set(groups_val)) == set(groups)

    # We use HDF5DataProvider here to determine input shape only.
    train_data = utils.keras.HDF5DataProvider(path, X_train.tolist(), batch_size=32, shuffle=True)
    input_shape, _ = train_data.get_data_shape()


    # >>>>>>>>>>>>>>>>>>>>>> MODEL:
    conv_layers = create_conv_blocks(
        input_shape=None,
        conv_blocks=[
            ConvBlockDef(
                number_of_kernels=32,
                kernel_size=51,
                pool_size=10
            ), # => 60 samples
            ConvBlockDef(
                number_of_kernels=64,
                kernel_size=6,
                pool_size=5
            ) # => 11 samples
        ],
        hidden_activation='relu'
    )
    fcn_layers = create_fcn_layers(
        input_shape=None, # determined by prev layer
        units=[1],
        hidden_activation='relu',
        output_activation='relu'
    )
    model = keras.models.Sequential(
        # reshape data to format acceptable by Conv1D (i.e. add feature channel)
        [keras.layers.core.Reshape(input_shape + (1,), input_shape=input_shape)] +
        conv_layers +
        [keras.layers.core.Flatten()] +
        fcn_layers
    )
    model.compile(
        optimizer='adam',
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )
    model.summary()
    model_wrapper = utils.keras.HDF5ModelWrapper(h5_file_path=path, model=model)
    history = model_wrapper.fit(
        X_train, y_train,
        batch_size=64, epochs=EPOCHS,
        validation_data=(X_val, y_val),
        shuffle=True
    )
    return PointEvalResults(
        best_estimator=model_wrapper,
        history=history
    )
    return model


if __name__ == "__main__":
    # Parse arguments.
    init_seeds()
    print("Started %s" % datetime.datetime.now().isoformat())
    print("Tensorflow version: %s" % tf.__version__)
    print("Keras version: %s" % keras.__version__)
    print("Scikit learn version: %s" % sklearn.__version__)
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", dest="path",
                        help="Path to the dataset.",
                        required=True)

    args = parser.parse_args()

    ids = None
    with h5py.File(args.path) as f:
        ids = f['ids'][:]
    x_idxs = np.array(range(len(ids)))
    y_idxs = np.array(range(len(ids)))

    outer_cv = sklearn.model_selection.GroupShuffleSplit(
        n_splits=no_test_splits,
        test_size=test_size,
        random_state=split_seed)
    print("Starting train/eval procedure...")

    scores = utils.sklearn.Multiscore(test_metrics, h5_file_path=args.path)
    results = utils.sklearn.cross_val_score_by_group(
        build_estimator_fn=functools.partial(create_model, path=args.path),
        X=x_idxs, y=y_idxs, groups=ids,
        cv=outer_cv,
        scores=scores
    )
    print("Train/eval finished.")
    print("Saving results to 'result' dir...")
    outer_results = utils.sklearn.get_outer_cv_results(results)

    with open(os.path.join("result", "outer_cv.pkl"), "wb") as f:
        pickle.dump(outer_results, f)

    with open("result/args.pkl", "wb") as f:
        pickle.dump(args, f)
    # save models to files
    for i, result in enumerate(results):
        model = result['build_result'].best_estimator.model
        history = result['build_result'].history
        model.save('result/model%d.h5' % i)
        with open("result/history%d.pkl" % i, "wb") as f:
            pickle.dump(history, f)
    exit(0)
