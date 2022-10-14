"""niworkflows.interfaces.fixes.FixHeaderApplyTransforms(),
niworkflows.interfaces.fixes.FixHeaderRegistration()"""


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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?output and whole func
@mark.task
@mark.annotate({"return": {"":}}) #?
def FixHeaderApplyTransforms(restrict_deformation:ty.List()):





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?output and whole func
@mark.task
@mark.annotate({"return": {"":}}) #?
def FixHeaderRegistration(transforms:ty.List[File]):



    