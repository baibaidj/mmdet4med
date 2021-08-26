from .custom_nn import CustomDatasetNN
from .custom_det import CustomDatasetDet

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