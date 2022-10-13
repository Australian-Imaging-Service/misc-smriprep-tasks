"""nitransforms tools interfaces for smri prep:
ConcatenateXFMs()"""

import os
from pathlib import Path
from nipype.interfaces.base import (
    TraitedSpec,
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    InputMultiObject,
    traits,
    isdefined,
)


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

XFM_FMT = {
    ".lta": "fs",
    ".txt": "itk",
    ".mat": "itk",
    ".tfm": "itk",
}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?
@mark.task
@mark.annotate({"return": {"out_xfm":File ,"out_inv":File}}) 
def ConcatenateXFMs(in_xfms:File, inverse:bool, out_fmt:str, reference:File, moving:File): #out_fmt:str?
    out_ext = "lta" if out_fmt == "fs" else "tfm"
    reference = reference if isdefined(reference) else None
    moving = moving if isdefined(moving) else None
    out_file = Path(os.getcwd()) / f"out_fwd.{out_ext}" #runtime.cwd?
    out_xfm = str(out_file)
    out_inv = None
    if inverse:
        out_inv = Path(os.getcwd()) / f"out_inv.{out_ext}" #runtime.cwd?
        out_inv = str(out_inv)

    concatenate_xfms(
        in_xfms,
        out_file,
        out_inv,
        reference=reference,
        moving=moving,
        fmt=out_fmt,
    )
    return out_xfm, out_inv


def concatenate_xfms(
    in_files, out_file, out_inv=None, reference=None, moving=None, fmt="itk"
):
    """Concatenate linear transforms."""
    from nitransforms.manip import TransformChain
    from nitransforms.linear import load as load_affine

    xfm = TransformChain(
        [load_affine(f, fmt=XFM_FMT[Path(f).suffix]) for f in in_files]
    ).asaffine()
    if reference is not None and not xfm.reference:
        xfm.reference = reference

    xfm.to_filename(out_file, moving=moving, fmt=fmt)

    if out_inv is not None:
        inv_xfm = ~xfm
        if moving is not None:
            inv_xfm.reference = moving
        inv_xfm.to_filename(out_inv, moving=reference, fmt=fmt)