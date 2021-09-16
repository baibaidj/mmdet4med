# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import subprocess

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file, reserve_keys = ('state_dict', 'meta')):
    checkpoint = torch.load(in_file, map_location='cpu')
    ckp_keys = list(checkpoint.keys())
    print('Checkpoint keys', checkpoint.keys())
    # remove optimizer for smaller file size
    # if 'optimizer' in checkpoint:
    #     del checkpoint['optimizer']
    for key in ckp_keys:
        if key not in reserve_keys:
            del checkpoint[key]
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    if torch.__version__ >= '1.6':
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file
    final_file = out_file_name + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)


if __name__ == '__main__':
    main()

# python tools/model_converters/publish_model.py /mnt/data4t/dejuns/ribfrac/model_save/v2.1.0/verse_instance_model_nnunet.pth  /mnt/data4t/dejuns/ribfrac/model_save/v2.2.0/verse_instance_model_nnunet.pth
# python tools/model_converters/publish_model.py /mnt/data4t/dejuns/ribfrac/model_save/v2.1.0/model_best-v1.ckpt  /mnt/data4t/dejuns/ribfrac/model_save/v2.2.0/nndet_v1_jqm
# python tools/model_converters/publish_model.py $repo_rt/$model_name/latest.pth $repo_rt/$model_name/publish.pth
