

from modules.deform_conv import DeformConv, _DeformConv, DeformConvPack
from modules.deform_conv import DeformConv_d, _DeformConv, DeformConvPack_d

# see test.py for usage
# DeformConvPack: using its own offsets
# DeformConv: using extra offsets
#     #  dimension = 'T' or 'H' or 'W' or any combination of these three letters
#     #  'T' represents the deformation in temporal dimension
#     #  'H' represents the deformation in height dimension
#     #  'W' represents the deformation in weigh dimension


