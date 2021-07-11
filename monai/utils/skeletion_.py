import json
import pickle
from functools import partial
from os import makedirs
from os.path import dirname, exists, join
from typing import List

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

makedirs = partial(makedirs, exist_ok=True)
from glob import glob


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.float):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def keep_largest_componet(arr):
    unq = np.unique(arr)[1:]
    coms = [ndimage.label(arr == i) for i in unq]
    # print([(c[0].shape, c[1]) for c in coms])
    idx = [[np.sum(c == i) for i in range(1, num + 1)] for c, num in coms]
    # print(idx)
    idx = [1 + np.argmax(s) for s in idx]
    # print(idx)
    coms = [c == i for i, (c, num) in zip(idx, coms)]
    # print([c.shape for c in coms])
    coms = np.stack([np.zeros(arr.shape)[np.newaxis]] +
                    [c[np.newaxis] for c in coms])
    coms = np.argmax(coms, axis=0).squeeze()

    return coms


def interjoint_paths_to_tree(skel):
    """
    Generate a tree struct by interjoint_paths from a skeleton.
    We assume that the root path is the one has the largest radius.
    """
    paths = skel.interjoint_paths(True)
    paths_set = [set(p) for p in paths]

    idxs = np.argsort(skel.radius)[::-1]
    print(f'max radius point, idx: {idxs[-1]},',
          f'radius: {skel.radius[idxs[-1]]},',
          f'vertice: {skel.vertices[idxs[-1]]}')

    # the max radius point may not sit on any path, why?
    for i in range(5):
        root = [od for od, pth in enumerate(paths_set) if idxs[i] in pth]
        if len(root) > 0:
            break
        print(f'Warning: try the {i+1}nd raius.')

    print(f'root candidate: {root}')
    if len(root) == 0:
        print('can not find root path.')
        return None

    # the root may connect with two paths.
    root = root[np.argmax([len(paths_set[i]) for i in root])]
    print(f'root path: {root}')

    matrix = np.array([[len(p.intersection(p0)) for p0 in paths_set]
                       for p in paths_set])
    tree = [{
        'depth': None,
        'mother': None,
        'child': np.argwhere(m)
    } for m in matrix]

    def loop(cur, mother, brother, depth):
        bra = tree[cur]
        bra['depth'] = depth

        if bra['mother'] is not None:
            return
        bra['mother'] = mother

        # exclude brother, mother and self
        child = np.array(bra['child'])
        exclude = brother
        if isinstance(brother, np.ndarray):
            exclude = brother.tolist()
        exclude = exclude + [mother, cur]
        idx = np.all([child != e for e in exclude], axis=0)
        bra['child'] = child[idx].tolist()
        # print(f'path {cur} find sub path: {bra["child"]}')

        depth += 1
        _ = [loop(c, cur, bra["child"], depth) for c in bra["child"]]

    loop(root, None, [], 1)

    return paths, tree


# from skimage.morphology import skeletonize_3d
# skel = skeletonize_3d(sitk.GetArrayFromImage(img1), )
def skeletion_wrapper(img):
    import kimimaro
    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img)
    skels = kimimaro.skeletonize(
        img,
        teasar_params={
            'scale': 1,
            'const': 4,  # physical units
            'pdrf_exponent': 8,
            #'pdrf_scale': 100000,
            #'soma_detection_threshold': 1100, # physical units
            #'soma_acceptance_threshold': 3500, # physical units
            #'soma_invalidation_scale': 1.0,
            #'soma_invalidation_const': 300, # physical units
            #'max_paths': 50, # default None
        },
        # object_ids=[1], # process only the specified labels
        # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
        # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
        # dust_threshold=1000, # skip connected components with fewer than this many voxels
        anisotropy=(1, 1, 1),  # default True
        fix_branching=True,  # default True
        fix_borders=True,  # default True
        fill_holes=False,  # default False
        fix_avocados=False,  # default False
        progress=True,  # default False, show progress bar
        parallel=8,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=
        100,  # how many skeletons to process before updating progress bar
    )
    return skels


def generate_skeletion_tree(skels):
    label_data = {}
    for key, skel in skels.items():
        paths, tree = interjoint_paths_to_tree(skel)
        vertices = skel.vertices.astype(np.int64)
        #vertices = [img.TransformIndexToPhysicalPoint(v.tolist()) for v in vertices]
        label_data[key] = {
            'path': paths,
            'tree': tree,
            'vertice': vertices,
            'radius': skel.radius
        }
    case_data = {'label': label_data}
    if hasattr(skel, 'affine'):
        case_data['affine'] = skel.affine

    return case_data


def skeletion_to_tree(input_name, output_name=None, overwriting=False):
    res = None
    try:
        if not overwriting and output_name and exists(output_name):
            print('process error, file exist: ', output_name)
            return res
        if isinstance(input_name, str):
            print(f'process case: {input_name}')
            with open(input_name, 'rb') as f:
                input_name = pickle.load(f)
        case_data = generate_skeletion_tree(input_name)
        if output_name is not None:
            makedirs(dirname(output_name))
            with open(output_name, 'w') as h:
                json.dump(case_data, h, cls=NumpyEncoder)
        res = case_data
    except Exception as e:
        print('process error: ', e)
    return res


def skeletonize_label(input_name, output_name=None, overwriting=False):
    if isinstance(input_name, str):
        print(f'process case: {input_name}')
    if output_name and not overwriting and exists(output_name):
        return True
    img = input_name
    if isinstance(input_name, str):
        img = sitk.ReadImage(input_name)
    arr = sitk.GetArrayFromImage(img)
    arr = keep_largest_componet(arr)
    skels = skeletion_wrapper(arr)
    for skel in skels.values():
        skel.affine = {
            'origin': img.GetOrigin(),
            'spacing': img.GetSpacing(),
            'direction': img.GetDirection()
        }

    if output_name is not None:
        makedirs(dirname(output_name))
        with open(output_name, 'wb') as f:
            pickle.dump(skels, f)
    return skels


def dump_skeletion_to_pkl(pkl_file: str,
                          skeletion_files: List[str],
                          keys: List[str],
                          overwriting: bool = False):
    if not overwriting and exists(pkl_file):
        raise IOError(f'The file is exist: {pkl_file}')
    assert len(skeletion_files) == len(
        keys), 'The length of nifti_files and keys is not equal!'
    data = {}
    for fname, key in zip(skeletion_files, keys):
        with open(fname, 'r') as f:
            case = json.load(f)
        data[key] = case
    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)
    return True
