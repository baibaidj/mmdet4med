
from .patching import *

def tile_case(self, images, update_remaining: bool = True) -> \
        Sequence[Dict[str, np.ndarray]]:
    """
    Create patches from whole patient for prediction

    Args:
        case: data of a single case
        update_remaining: properties from case which are not tiles
            are saved into all patches

    Returns:
        Sequence[Dict[str, np.ndarray]]: extracted crops from case
            and added new key:
                `tile_origin`: Sequence[int] offset of tile relative
                    to case origin
    """
    dshape = images.shape
    overlap = [int(c * self.overlap) for c in self.crop_size] # stride in jianqiang's case
    crops = create_grid(
        cshape=self.crop_size,
        dshape=dshape[1:],
        overlap=overlap,
        mode=self.grid_mode,
        )

    tiles = []
    for crop in crops:
        try:
            # try selected extraction mode
            tile = {key: save_get_crop(case[key], crop, mode=self.save_get_mode)[0]
                    for key in self.tile_keys}
            _, tile["tile_origin"], tile["crop"] = save_get_crop(
                case[self.tile_keys[0]], crop, mode=self.save_get_mode)
        except RuntimeError:
            # fallback to symmetric
            logger.warning("Path size is bigger than whole case, padding case to match patch size")
            tile = {key: save_get_crop(case[key], crop, mode="symmetric")[0]
                    for key in self.tile_keys}
            _, tile["tile_origin"], tile["crop"] = save_get_crop(
                case[self.tile_keys[0]], crop, mode="symmetric")

        if update_remaining:
            tile.update({key: item for key, item in case.items()
                            if key not in self.tile_keys})
        tiles.append(tile)
    return tiles