from .custom_nn import CustomDatasetNN
from .custom_det import CustomDatasetDet
from .custom_seg import CustomDatasetMonai
from .transform4med.io4med import os, osp, load_string_list, np

from .builder import DATASETS

@DATASETS.register_module()
class RibFractureNN(CustomDatasetNN):

    CLASSES = ('ribfrac', )

    def __init__(self, *args, **kwargs):
        super(RibFractureNN, self).__init__(*args, **kwargs)


@DATASETS.register_module()
class RibFractureDet(CustomDatasetDet):

    CLASSES = ('ribfrac', )

    def __init__(self, *args, **kwargs):
        super(RibFractureDet, self).__init__(*args, **kwargs)



@DATASETS.register_module()
class RibFractureDet3cls(CustomDatasetDet):

    CLASSES = ('broke', 'buckle', 'old')

    def __init__(self, *args, **kwargs):
        super(RibFractureDet3cls, self).__init__(*args, **kwargs)




@DATASETS.register_module()
class AllCTDataset(CustomDatasetMonai):
    """Pneumonia dataset.

    The ``img_suffix`` is fixed to '_leftImg8bewdcfit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    CLASSES = ('bg', 'artery')
    def __init__(self, *args, exclude_pids = None, **kwargs):
        super(AllCTDataset, self).__init__(*args, exclude_pids = exclude_pids, **kwargs)

        self.gt_seg_maps = None
        self.flag = np.ones(len(self), dtype=np.uint8)

    
    def _img_list2dataset(self, data_folder:str, **kwags):
        """
        
        return 
            file_list : [{'image' : img_path, 'label' : label_path}, ...]
        """
        # a = [print(self.map_key(k)) for k in keys]
        js_fp = os.path.join(data_folder, self.json_filename)
        if not osp.exists(js_fp): return []
        image_fps = load_string_list(js_fp)
        pid2pathpairs = []
        for ifp  in image_fps:
            cid = ifp.split(os.sep)[-1].split('.')[0]
            this_pair = {'cid': cid, 'img': ifp}
            pid2pathpairs.append(this_pair)
        pathpairs_orderd = sorted(pid2pathpairs, key = lambda x: x['cid'])
        print(f'[RawCT] {len(pathpairs_orderd)} samples')
        return pathpairs_orderd
    