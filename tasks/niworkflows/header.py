"""header tools interfaces for smri prep:
CopyXForm(), ValidateImage()"""

import os
import shutil
from textwrap import indent
import numpy as np
import nibabel as nb
import transforms3d

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits,
    File,
    TraitedSpec,
    BaseInterfaceInputSpec,
    SimpleInterface,
    DynamicTraitedSpec,
)
from nipype.interfaces.io import add_traits
from ..utils.images import _copyxform
from .. import __version__


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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~??????many
@mark.task
@mark.annotate({"return": {"": }}) #???
def CopyXForm(hdr_file: File): #can I use DynamicTraitedSpec?
    for f in SimpleInterface._fields: #self._fields?????
        in_files = getattr(SimpleInterface.inputs, f)
        f = []
        if isinstance(in_files, str):
            in_files = [in_files]
        for in_file in in_files:
            out_name = fname_presuffix(
                in_file, suffix="_xform", newpath=os.getcwd()
            ) #newpath=runtime.cwd???
            # Copy and replace header
            shutil.copy(in_file, out_name)
            _copyxform(
                hdr_file,
                out_name,
                message="CopyXForm (niworkflows v%s)" % __version__,
            )
            f.append(out_name)

        # Flatten out one-element lists
        if len(f) == 1:
            f = f[0]

    default = self._results.pop("in_file", None) ###?default = self._results.pop("in_file", None)
    if default:
        out_file = default
    return runtime #####?????????

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?

@mark.task
@mark.annotate({"return": {"out_file": File, "out_report": File }}) 
def ValidateImage(in_file: File): #can I use DynamicTraitedSpec?
    img = nb.load(in_file)
    out_report = os.path.join(os.getcwd(), "report.html") #runtime.cwd

    # Retrieve xform codes
    sform_code = int(img.header._structarr["sform_code"])
    qform_code = int(img.header._structarr["qform_code"])

    # Check qform is valid
    valid_qform = False
    try:
        qform = img.get_qform()
        valid_qform = True
    except ValueError:
        pass

    sform = img.get_sform()
    if np.linalg.det(sform) == 0:
        valid_sform = False
    else:
        RZS = sform[:3, :3]
        zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
        valid_sform = np.allclose(zooms, img.header.get_zooms()[:3])

    # Matching affines
    matching_affines = valid_qform and np.allclose(qform, sform)

    # Both match, qform valid (implicit with match), codes okay -> do nothing, empty report
    if matching_affines and qform_code > 0 and sform_code > 0:
        out_file = in_file
        open(out_report, "w").close()
        out_report = out_report
        return out_file, out_report

    # A new file will be written
    out_fname = fname_presuffix(
        in_file, suffix="_valid", newpath=os.getcwd()
    )
    out_file = out_fname

    # Row 2:
    if valid_qform and qform_code > 0 and (sform_code == 0 or not valid_sform):
        img.set_sform(qform, qform_code)
        warning_txt = "Note on orientation: sform matrix set"
        description = """\
<p class="elem-desc">The sform has been copied from qform.</p>
"""
    # Rows 3-4:
    # Note: if qform is not valid, matching_affines is False
    elif (valid_sform and sform_code > 0) and (
        not matching_affines or qform_code == 0
    ):
        img.set_qform(sform, sform_code)
        new_qform = img.get_qform()
        if valid_qform:
            # False alarm - the difference is due to precision loss of qform
            if np.allclose(new_qform, qform) and qform_code > 0:
                out_file = in_file
                open(out_report, "w").close()
                out_report = out_report
                return out_file, out_report
            # Replacing an existing, valid qform. Report magnitude of change.
            diff = np.linalg.inv(qform) @ new_qform
            trans, rot, _, _ = transforms3d.affines.decompose44(diff)
            angle = transforms3d.axangles.mat2axangle(rot)[1]
            xyz_unit = img.header.get_xyzt_units()[0]
            if xyz_unit == "unknown":
                xyz_unit = "mm"

            total_trans = np.sqrt(
                np.sum(trans * trans)
            )  # Add angle and total_trans to report
            warning_txt = "Note on orientation: qform matrix overwritten"
            description = f"""\
<p class="elem-desc">
The qform has been copied from sform.
The difference in angle is {angle:.02g} radians.
The difference in translation is {total_trans:.02g}{xyz_unit}.
</p>
"""
        elif qform_code > 0:
            # qform code indicates the qform is supposed to be valid. Use more stridency.
            warning_txt = "WARNING - Invalid qform information"
            description = """\
<p class="elem-desc">
The qform matrix found in the file header is invalid.
The qform has been copied from sform.
Checking the original qform information from the data produced
by the scanner is advised.
</p>
"""
        else:  # qform_code == 0
            # qform is not expected to be valids. Simple note.
            warning_txt = "Note on orientation: qform matrix overwritten"
            description = (
                '<p class="elem-desc">The qform has been copied from sform.</p>'
            )
    # Rows 5-6:
    else:
        affine = img.header.get_base_affine()
        img.set_sform(affine, nb.nifti1.xform_codes["scanner"])
        img.set_qform(affine, nb.nifti1.xform_codes["scanner"])
        warning_txt = "WARNING - Missing orientation information"
        description = """\
<p class="elem-desc">
FMRIPREP could not retrieve orientation information from the image header.
The qform and sform matrices have been set to a default, LAS-oriented affine.
Analyses of this dataset MAY BE INVALID.
</p>
"""
    snippet = '<h3 class="elem-title">%s</h3>\n%s\n' % (warning_txt, description)
    # Store new file and report
    img.to_filename(out_fname)
    with open(out_report, "w") as fobj:
        fobj.write(indent(snippet, "\t" * 3))

    out_report = out_report
    out_file = out_fname ###? Added by me 
    return out_file, out_report

