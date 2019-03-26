import h5py
import glob
import os
import re
import argparse
import numpy as np


def preprocess_line(line, decimate):
    # 1. trim leading values, which are close to zero
    eps = 1e-10
    z = np.where(np.abs(line) > eps)[0]
    line = line[z[0]:]
    # 2. decimate line
    if decimate != 1:
        line = line[::decimate]
    return line


def standardize(x):
    return (x-np.mean(x))/np.std(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Chunks given dataset into multiple parts along T axis, saves it as h5py file.""")

    parser.add_argument("--dataset_dir", dest="dataset_dir",
                        help="Path to the dataset; this script will look for a files named att{xx}.mat, and treat {xx} as an attenuation times 1e-1.",
                        required=True)
    parser.add_argument("--chunk_size", dest="chunk_size", type=int,
                        help="Chunk size, in number of samples.",
                        required=True)
    parser.add_argument("--stride", dest="stride", type=int,
                        help="The distance between two slicing windows, in number of samples.",
                        required=True)
    parser.add_argument("--output_file", dest="output_file",
                        help="Name of the output HDF5 file.",
                        required=True)
    parser.add_argument("--decimate", dest="decimate", type=int, default=1,
                        help="Decimation factor. 1 means no decimation; 2 means removing half of all samples.")
    parser.add_argument("--standardize", dest="standardize", type=int, default=1,
                        help="If equal to 1, performs chunk-wise standarization (from each chunk-mean(chunk)/std(chunk))")
    args = parser.parse_args()
    assert args.chunk_size >= args.stride

    filenames = sorted(glob.glob(os.path.join(args.dataset_dir, "att*.mat")))

    # Calculate shape of the X and y (required to determine hdf5 array shape).
    no_samples = 0
    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            rf = f['rf']
            for line_n in range(rf.shape[0]):
                line = preprocess_line(rf[line_n, :], decimate=args.decimate)
                no_steps = line.shape[0]-args.chunk_size+1
                no_samples += (no_steps-1)//args.stride+1

    # Create THE FILE.
    print("Number of samples %d " % no_samples)
    print("Creating THE FILE %s..." % args.output_file)
    with h5py.File(os.path.join(args.dataset_dir, args.output_file), 'w') as output_f:
        X = output_f.create_dataset("X", (no_samples, args.chunk_size), dtype="float64")
        y = output_f.create_dataset("y", (no_samples,), dtype="float64")
        ids = output_f.create_dataset("ids", (no_samples,), dtype="i") # scanline number
        i = 0
        scanline_id = 0
        for filename in filenames:
            # Extract alpha coeff from filename.
            _, name = os.path.split(filename)
            alpha = int(re.search("att(\d+)\.mat", name).group(1))*1e-1
            with h5py.File(filename, 'r') as input_f:
                rf = input_f['rf']
                for line_n in range(rf.shape[0]):
                    line = preprocess_line(rf[line_n, :], decimate=args.decimate)
                    t = 0
                    chunks = []
                    while t+args.chunk_size < line.shape[0]:
                        chunk = line[t:t+args.chunk_size]
                        if args.standardize:
                            chunk = standardize(chunk)
                        chunks.append(chunk)
                        t += args.stride
                    chunks = np.stack(chunks, axis=0)
                    chunk_ys = np.array([alpha]*chunks.shape[0])
                    chunk_ids = np.array([scanline_id]*chunks.shape[0])
                    # save chunks
                    X[i:i+chunks.shape[0],:] = chunks
                    y[i:i+chunks.shape[0]] = chunk_ys
                    ids[i:i+chunks.shape[0]] = chunk_ids
                    # update vals
                    i += chunks.shape[0]
                    scanline_id += 1
        # save metadata
        X.attrs['chunk_size'] = args.chunk_size
        X.attrs['stride'] = args.stride
        X.attrs['decimate'] = args.decimate
        X.attrs['standardize'] = args.standardize






