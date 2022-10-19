"""niworkflows.interfaces.fixes.FixHeaderApplyTransforms(),
niworkflows.interfaces.fixes.FixHeaderRegistration()"""


import os
from pathlib import Path
import nibabel as nb
import numpy as np
import logging
import attrs
from pydra import mark
import typing as ty
from pydra.engine.specs import (
    ShellSpec,
    ShellOutSpec,
    File,
    Path,
    MultiInputFile,
    Directory,
    SpecInfo,
    MultiInputFile,
    MultiInputObj,
)

logger = logging.getLogger("pydra-niworkflows")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?output and whole func

fixed_apply_transform_inputs = [
    (
        "transforms",
        MultiInputFile,
        {
            "argstr": "%s",
            "mandatory":True,
            "desc": "transform files: will be applied in reverse order. For "
            "example, the last specified transform will be applied first.",
        }
    )
]

FixHeaderApplyTransforms_input_spec = SpecInfo(
    name="FixHeaderApplyTransforms_Input", fields=fixed_apply_transform_inputs,
    bases=(ApplyTransforms_input_spec,)
)


class FixHeaderApplyTransforms(ApplyTransforms):

    input_spec = FixHeaderApplyTransforms_input_spec

    def _run_task(self):
        super()._run_task()

        _copyxform(
            self.inputs.reference_image,
            os.path.abspath(self.outputs.output_image),
            message="%s (niworkflows v%s)" % (self.__class__.__name__, "unknown"),  # __version__
        )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?output and whole func

class FixHeaderRegistration(Registration):
    pass




def _copyxform(ref_image, out_image, message=None):
    # Read in reference and output
    # Use mmap=False because we will be overwriting the output image
    resampled = nb.load(out_image, mmap=False)
    orig = nb.load(ref_image)

    if not np.allclose(orig.affine, resampled.affine):

        logger.debug(
            "Affines of input and reference images do not match, "
            "FMRIPREP will set the reference image headers. "
            "Please, check that the x-form matrices of the input dataset"
            "are correct and manually verify the alignment of results."
        )

    # Copy xform infos
    qform, qform_code = orig.header.get_qform(coded=True)
    sform, sform_code = orig.header.get_sform(coded=True)
    header = resampled.header.copy()
    header.set_qform(qform, int(qform_code))
    header.set_sform(sform, int(sform_code))
    header["descrip"] = "xform matrices modified by %s." % (message or "(unknown)")

    newimg = resampled.__class__(resampled.dataobj, orig.affine, header)
    newimg.to_filename(out_image)