from nndet.io.load import load_json
import os
import shutil
from pathlib import Path

import pandas as pd

from nndet.io import save_json
from nndet.utils.check import env_guard
from nndet.utils.info import maybe_verbose_iterable
import numpy as np
import os.path as osp


def create(
    image_source: Path,
    label_source: Path,
    image_target_dir: Path,
    label_target_dir: Path,
    df: pd.DataFrame,
    fg_only: bool = False,
    ):
    image_target_dir.mkdir(parents=True, exist_ok=True)
    label_target_dir.mkdir(parents=True, exist_ok=True)

    case_id = image_source.stem.rsplit('_image', 1)[0]
    case_id_check = label_source.stem.rsplit('_instance', 1)[0]
    assert case_id == case_id_check, f"case ids not matching, found image {case_id} and label {case_id_check}"

    df_case = df.loc[df['public_id'] == case_id]
    instances = {}
    for row in df_case.itertuples():
        _cls = int(row.label_code)
        if _cls == 0:   # background has label code 0 and lab id 0
            continue

        if fg_only:
            _cls = 1
        elif _cls == -1:
            _cls = 5

        instances[str(row.label_id)] = _cls - 1  # class range from 0 - 4 // if fg only 0
        assert 0 < _cls < 6, f"Something strange happened {_cls}"
    save_json({"instances": instances}, label_target_dir / f"{case_id}.json")

    shutil.copy2(image_source, image_target_dir / f"{case_id}_0000.nii.gz")
    shutil.copy2(label_source, label_target_dir / f"{case_id}.nii.gz")


@env_guard
def main():
    det_data_dir = Path(os.getenv('det_data'))
    task_data_dir = det_data_dir / "Task020FG_RibFrac"
    source_data_dir = task_data_dir / "raw"

    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    image_paths, label_paths, df = prepare_data_info()
    # image_paths = list((source_data_dir / "imagesTr").glob("*.nii.gz"))
    # image_paths.sort()
    # label_paths = list((source_data_dir / "labelsTr").glob("*.nii.gz"))
    # label_paths.sort()

    print(f"Found {len(image_paths)} data files and {len(label_paths)} label files.")
    assert len(image_paths) == len(label_paths)

    meta = {
        "name": "RibFracFG",
        "task": "Task020FG_RibFrac",
        "target_class": None,
        "test_labels": False,
        "labels": {"0": "fracture"}, # since we are running FG vs BG this is not completely correct
        "modalities": {"0": "CT"},
        "dim": 3,
    }
    save_json(meta, task_data_dir / "dataset.json")

    for ip, lp in maybe_verbose_iterable(list(zip(image_paths, label_paths))):
        create(image_source=ip,
               label_source=lp,
               image_target_dir=target_data_dir,
               label_target_dir=target_label_dir,
               df=df,
               fg_only=True,
               )

def prepare_data_info(store_root = Path('/data/dejuns/ribfrac/processed/plan_fracture_instance'), 
                    ):

    taskids = [ 
                'KY_B0_100',
                'KY_B1_30', 
                'KY_B2_20',  # # B0: 1391448_20190715/1391448_20190715
                'KY_B3_20', 
                'KY_B4_50',
                'KY_B5_74',
                'KY_B6_48', 
                'pub_ribfrac20'
                ]
    mask_fn = 'instance'
    img_fn, info_fn = 'image', 'ins2cls'
    # store_root = Path('/data/dejuns/ribfrac/processed/plan_rib_crop')
    key2dtypes = {'image' : np.int16, 'fracture': np.uint8, 'ribs' : np.uint8, 'distance' : float} # , 'label': np.uint8
    # taskid = taskids[0]
    image_paths, label_paths, roi_info_list = [], [], []
    # public_id,label_id,label_code
    for i, taskid in enumerate(taskids):
        taskid_dst = taskid
        dst_dir = store_root/taskid_dst
        print(f' {i}th Taskid {taskid} ')
        case_dirs_raw = [a for a in os.listdir(dst_dir) if osp.isdir(dst_dir/a) and '.' not in a]
        # pdb.set_trace()
        case_dirs = sorted(case_dirs_raw ) #key = lambda x: int(x[7:]) 
        print(f'TASKID {taskid} contains {len(case_dirs)} cases')
        for ci, case_dir in enumerate(case_dirs):
            image_fp = dst_dir/f'{case_dir}/{case_dir}_{img_fn}.nii.gz'
            label_fp = dst_dir/f'{case_dir}/{case_dir}_{mask_fn}.nii.gz'
            roi_info_fp = dst_dir/f'{case_dir}/{case_dir}_{info_fn}.json' 
            if not (osp.exists(image_fp) and osp.exists(label_fp)):
                print(case_dir, 'not exists')
                continue
            image_paths.append(image_fp)
            label_paths.append(label_fp)
            rois_info = load_json(roi_info_fp)
            for roi in rois_info:
                this_roi = {'public_id' : case_dir, 
                            'label_id': roi['instance'], 
                            'label_code': roi['class']}
                roi_info_list.append(this_roi)

    roi_info_tb = pd.DataFrame(roi_info_list)
    roi_info_tb.to_csv(store_root/'fracture_ins2class_pubformat.csv', index = False)

    return image_paths, label_paths, roi_info_tb

# roi_info = {'instance' : ix + 1, 
#             'bbox': roi_bbox, 
#             'class': -1 if roi_cls == 66535 else roi_cls , 
#             'center' : roi_center}

if __name__ == '__main__':
    main()
