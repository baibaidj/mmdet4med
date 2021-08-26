_base_ = [
    '../models/retinanet_r18_4l8c_vnet_3d_surefg.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True) #Sync
