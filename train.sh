#!/bin/bash
set -e

if [ -z "${PROJECT_PATH}" ]; then
    echo "PROJECT_PATH is uneset/empty, set it to path, where the project is placed. "
    exit 1
fi
export PYTHONPATH=${PROJECT_PATH}:$PYTHONPATH

eval_name="$1"
if [ "${eval_name}" != '.' ] ; then
    echo "${@}" > "evals/${eval_name}/input.log"
    echo "Executing eval '${eval_name}'..."
    current_dir=$(pwd)
    cd "evals/${eval_name}"
fi
python3 "train.py" "${@:2}"
if [ "${eval_name}" != '.' ] ; then
    cd $current_dir # back to previous dir
fi
echo "Eval '${eval_name}' finished successfully."
