"""some  niworkflows.interfaces.reportlets tools interfaces for smri prep:
reportlets.masks.ROIsPlot(), reportlets.registration.SimpleBeforeAfterRPT()"""

from nipype.interfaces.base import (
    File,
    isdefined,
)



import os
from pathlib import Path
import nibabel as nb
import numpy as np
from pydra import mark
import typing as ty
from pydra.engine.specs import (
    ShellSpec,
    ShellOutSpec,
    File,
    Path,
    Directory,
    SpecInfo,
    MultiInputFile,
    MultiInputObj,
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?
@mark.task
@mark.annotate({"return": {"":}}) #?
def ROIsPlot(in_file:File, in_rois: ty.List[str] , in_mask:File, masked:bool, colors:ty.List[str] ,levels:ty.List[float] , mask_color:str): #in_rois: ty.List[str]?--, out_report: , compress_report: should I to input? they are not available in the inspec

    from seaborn import color_palette
    from niworkflows.viz.utils import plot_segs, compose_view

    seg_files = in_rois
    mask_file = None if not isdefined(in_mask) else in_mask

    # Remove trait decoration and replace None with []
    levels = [level for level in levels or []]
    colors = [c for c in colors or []]

    if len(seg_files) == 1:  # in_rois is a segmentation
        nsegs = len(levels)
        if nsegs == 0:
            levels = np.unique(
                np.round(nb.load(seg_files[0]).get_fdata(dtype="float32"))
            )
            levels = (levels[levels > 0] - 0.5).tolist()
            nsegs = len(levels)

        levels = [levels]
        missing = nsegs - len(colors)
        if missing > 0:
            colors = colors + color_palette("husl", missing)
        colors = [colors]
    else:  # in_rois is a list of masks
        nsegs = len(seg_files)
        levels = [[0.5]] * nsegs
        missing = nsegs - len(colors)
        if missing > 0:
            colors = [[c] for c in colors + color_palette("husl", missing)]

    if mask_file:
        seg_files.insert(0, mask_file)
        if levels:
            levels.insert(0, [0.5])
        colors.insert(0, [mask_color])
        nsegs += 1

    _out_report = os.path.abspath(out_report)
    compose_view(
        plot_segs(
            image_nii=in_file,
            seg_niis=seg_files,
            bbox_nii=mask_file,
            levels=levels,
            colors=colors,
            out_file=out_report, #? was not in the input spec
            masked=masked,
            compress=compress_report, #? was not in the input spec
        ),
        fg_svgs=None,
        out_file=_out_report,
    )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~what should  do if diff def inside the class? --- what should I do with the names of the defs?
@mark.task
@mark.annotate({"return": {"":}}) 
def SimpleBeforeAfterRPT(before:File, after:File , wm_seg:File, before_label:str, after_label:str,dismiss_affine:bool): 
    """ there is not inner interface to run """
    import logging
    __packagename__ = "niworkflows"
    NIWORKFLOWS_LOG = logging.getLogger(__packagename__)
    _fixed_image_label = after_label
    _moving_image_label = before_label
    _fixed_image = after
    _moving_image = before
    _contour = wm_seg if isdefined(wm_seg) else None
    _dismiss_affine = dismiss_affine
    NIWORKFLOWS_LOG.info(
        "Report - setting before (%s) and after (%s) images",
        _fixed_image,
        _moving_image,
    )

    return super(SimpleBeforeAfterRPT, self)._post_run_hook(runtime) ##????????

    
