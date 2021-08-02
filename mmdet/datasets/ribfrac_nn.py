from .custom_nn import CustomDatasetNN

from .builder import DATASETS


@DATASETS.register_module()
class RibFractureNN(CustomDatasetNN):

    CLASSES = ('ribfrac', )

    def __init__(self, *args, **kwargs):
        super(RibFractureNN, self).__init__(*args, **kwargs)