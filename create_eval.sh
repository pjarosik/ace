#!/bin/bash
set -e

eval_name="$1"
if [ "${eval_name}" != '.' ] ; then
    if [ -d "evals/${eval_name}/result" ] ; then
        echo "Eval with given name already exists!"
        exit -1
    fi
    echo "Copying template to '${eval_name}'..."
    mkdir -p "evals/${eval_name}/result"
    # here put files, which are required for one evaluation
    cp {train.py,data.ipynb,test_results.ipynb} "evals/${eval_name}/"
fi
