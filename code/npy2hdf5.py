import h5py as h5
import numpy as np
import os.path as osp
import glob
import tqdm
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feat_folder',
        default='/home/pardogl/datasets/movies/youtube/',
        type=str,
        help='Path to folder contain all video folders')
    parser.add_argument(
        '--out_path',
        default='/home/pardogl/datasets/movies/',
        type=str)
    parser.add_argument(
        '--suffix_features',
        type=str, help='Suffix of npy features name')
    parser.add_argument(
        '--out_name',
        default='/home/pardogl/datasets/movies/',
        type=str)
    return parser.parse_args()


def npy2h5(feat_folder, out_path, suffix_features, out_name):
    features_path = (f'{feat_folder}/*')
    videos = glob.glob(f'{features_path}/*{suffix_features}.npy')
    print(f'{len(videos)} features found')
    features = []
    names = []
    for video in tqdm.tqdm(videos):
        feature = np.load(open(video,'rb'))
        features.append(feature)
        name = osp.basename(video).replace(f'{suffix_features}.npy','')
        names.append(name)

    print('Saving hdf5 file')
    with h5.File(f'{out_path}/{out_name}.h5','w') as f:
        for name, feature in tqdm.tqdm(zip(names, features), total=len(names)):
            f.create_dataset(name, data=feature, chunks=True)

if __name__ == "__main__":
    args = get_arguments()
    npy2h5(args.feat_folder, args.out_path, args.suffix_features, args.out_name)
