"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

from typing import Any, Callable, Sequence, Union

import torch

from monai.inferers.inferer import Inferer
from monai.inferers.utils_ import sliding_window_multi_pyramid_inference, sliding_window_detection
from monai.utils import BlendMode, PytorchPadMode
import numpy as np

__all__ = ["SlidingWindowMultiPyramidInferer", "SlidingWindowDetectionInferer"]


class SlidingWindowMultiPyramidInferer(Inferer):
    """
    Sliding window method for model inference,
    with `sw_batch_size` windows for every model.forward().

    Args:
        roi_size: the window size to execute SlidingWindow evaluation.
            If it has non-positive components, the corresponding `inputs` size will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/nn.functional.html#pad
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.

    Note:
        ``sw_batch_size`` denotes the max number of windows per network inference iteration,
        not the batch size of inputs.

    """

    def __init__(
            self,
            roi_size: Union[Sequence[int], int],
            sw_batch_size: int = 1,
            overlap: float = 0.25,
            mode: Union[BlendMode, str] = BlendMode.CONSTANT,
            sigma_scale: Union[Sequence[float], float] = 0.125,
            padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
            cval: float = 0.0,
            pyramid_scale: Union[Sequence[float], float] = (1.0,),
            sw_device: Union[torch.device, str, None] = None,
            device: Union[torch.device, str, None] = None,
    ) -> None:
        Inferer.__init__(self)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.pyramid_scale = pyramid_scale
        self.sw_device = sw_device
        self.device = device

    def __call__(
            self,
            inputs: torch.Tensor,
            network: Callable[..., torch.Tensor],
            *args: Any,
            **kwargs: Any,
    ) -> torch.Tensor:
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        return sliding_window_multi_pyramid_inference(
            inputs,
            self.roi_size,
            self.sw_batch_size,
            network,
            self.overlap,
            self.mode,
            self.sigma_scale,
            self.padding_mode,
            self.cval,
            self.pyramid_scale,
            self.sw_device,
            self.device,
            *args,
            **kwargs,
        )


class SlidingWindowDetectionInferer(Inferer):
    """
    Sliding window method for model inference for detection method,
    with `sw_batch_size` windows for every model.forward().

    Args:
        poster: refine detection bbox
        roi_size: the window size to execute SlidingWindow evaluation.
            If it has non-positive components, the corresponding `inputs` size will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/nn.functional.html#pad
        cval: fill value for 'constant' padding mode. Default: 0
        dim: 2D or 3D
        model_min_confidence: minimum confidence value for detected bounding box
        detection_nms_threshold: nms threshold, default as 1e-5 means not allow overlapping among bounding boxes
        model_max_instances_per_batch_element: maximum number of detected bounding box instances per batch image
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.

    Note:
        ``sw_batch_size`` denotes the max number of windows per network inference iteration,
        not the batch size of inputs.

    """

    def __init__(
            self,
            anchors,
            roi_size: Union[Sequence[int], int],
            sw_batch_size: int = 1,
            overlap: float = 0.25,
            padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
            cval: float = 0.0,
            dim: int = 3,
            rpn_bbox_std_dev=np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]),
            model_min_confidence: float = 0.1,
            detection_nms_threshold: float = 1e-5,
            model_max_instances_per_batch_element: int = 2000,
            sw_device: Union[torch.device, str, None] = None,
            device: Union[torch.device, str, None] = None,
    ) -> None:
        Inferer.__init__(self)
        self.anchors = anchors
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.padding_mode = padding_mode
        self.cval = cval
        self.dim = dim
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.model_min_confidence = model_min_confidence
        self.detection_nms_threshold = detection_nms_threshold
        self.model_max_instances_per_batch_element = model_max_instances_per_batch_element
        self.sw_device = sw_device
        self.device = device

    def __call__(
            self,
            inputs: torch.Tensor,
            network: Callable[..., torch.Tensor],
            *args: Any,
            **kwargs: Any,
    ) -> torch.Tensor:
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        return sliding_window_detection(
            self.anchors,
            inputs,
            self.roi_size,
            self.sw_batch_size,
            network,
            self.overlap,
            self.padding_mode,
            self.cval,
            self.dim,
            self.rpn_bbox_std_dev,
            self.model_min_confidence,
            self.detection_nms_threshold,
            self.model_max_instances_per_batch_element,
            self.sw_device,
            self.device,
            *args,
            **kwargs,
        )
