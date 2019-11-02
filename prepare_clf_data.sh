#!/bin/bash
# Creates clf data to compare 01 - 02, 01-03, etc...
set -e

dir=$1
task=$2
if [ -z $3 ] ; then
    filename_prefix=1cm_dec2
    chunk_size=650
    stride=325
else
    filename_prefix=$3
    chunk_size=$4
    stride=$5
fi

for i in {2..15};
do
    i_str=$(printf "%02d" $i)
    output_dir="${dir}/simulated_01${i_str}"
    output_file="${filename_prefix}_${task}.hdf5"
    echo "Output dir: ${output_dir}"
    if [ ! -d "$output_dir" ] ; then
        mkdir -p ${output_dir}
        cp "${dir}/simulated/att01.mat" "${output_dir}"
        cp "${dir}/simulated/att${i_str}.mat" "${output_dir}"
    fi

    if [ -f "${output_dir}/${output_file}" ] ; then
        echo "File already exists: ${output_dir}/${output_file}. Will be removed."
        rm "${output_dir}/${output_file}"
    fi
    python prepare_data.py --dataset_dir=${output_dir} --chunk_size=${chunk_size} --stride=${stride} --decimate=2 --standardize=1 --output_file=${output_file} --task="${task}"
done
