"""FreeSurfer tools interfaces for smri prep:
FSDetectInputs(), FSInjectBrainExtracted(), MakeMidthickness(),
PatchedLTAConvert(), PatchedRobustRegister(), RefineBrainMask()"""

import os.path as op
from pathlib import Path
import nibabel as nb
import numpy as np

from nipype.utils.filemanip import copyfile, filename_to_list, fname_presuffix
from nipype.interfaces.base import (
    isdefined,
    InputMultiPath,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    Directory,
)
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.base import SimpleInterface
from nipype.interfaces.freesurfer.preprocess import ConcatenateLTA, RobustRegister
from nipype.interfaces.freesurfer.utils import LTAConvert
from .reportlets.registration import BBRegisterRPT, MRICoregRPT

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ok
@mark.task
@mark.annotate({"return": {"t2w": File, "use_t2w": bool, "flair": File ,"use_flair": bool , "hires": bool ,"mris_inflate": str}})
def FSDetectInputs(t1w_list,t2w_list,flair_list,hires_enabled):
    t2w, flair, hires, mris_inflate = detect_inputs(
            t1w_list,
            t2w_list=t2w_list if isdefined(t2w_list) else None,
            flair_list=flair_list
            if isdefined(flair_list)
            else None,
            hires_enabled=hires_enabled,
    )

    use_t2w = t2w is not None
    if use_t2w:
        t2w = t2w

    use_flair = flair is not None
    if use_flair:
        flair = flair

    if hires:
        mris_inflate = mris_inflate

    return t2w, use_t2w, flair, use_flair, hires, mris_inflate


def detect_inputs(t1w_list, t2w_list=None, flair_list=None, hires_enabled=True):
    t1w_list = filename_to_list(t1w_list)
    t2w_list = filename_to_list(t2w_list) if t2w_list is not None else []
    flair_list = filename_to_list(flair_list) if flair_list is not None else []
    t1w_ref = nb.load(t1w_list[0])
    # Use high resolution preprocessing if voxel size < 1.0mm
    # Tolerance of 0.05mm requires that rounds down to 0.9mm or lower
    hires = hires_enabled and max(t1w_ref.header.get_zooms()) < 1 - 0.05

    t2w = None
    if t2w_list and max(nb.load(t2w_list[0]).header.get_zooms()) < 1.2:
        t2w = t2w_list[0]

    # Prefer T2w to FLAIR if both present and T2w satisfies
    flair = None
    if flair_list and not t2w and max(nb.load(flair_list[0]).header.get_zooms()) < 1.2:
        flair = flair_list[0]

    # https://surfer.nmr.mgh.harvard.edu/fswiki/SubmillimeterRecon
    mris_inflate = "-n 50" if hires else None
    return (t2w, flair, hires, mris_inflate)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ok

@mark.task
@mark.annotate({"return": {"subjects_dir": Directory , "subject_id": str}})
def FSInjectBrainExtracted(subjects_dir: Directory, subject_id: str, in_brain: File):
    subjects_dir, subject_id = inject_skullstripped(
        subjects_dir, subject_id, in_brain
    )
    subjects_dir = subjects_dir
    subject_id = subject_id
    return subjects_dir, subject_id


def inject_skullstripped(subjects_dir, subject_id, skullstripped):
    from nilearn.image import resample_to_img, new_img_like

    mridir = op.join(subjects_dir, subject_id, "mri")
    t1 = op.join(mridir, "T1.mgz")
    bm_auto = op.join(mridir, "brainmask.auto.mgz")
    bm = op.join(mridir, "brainmask.mgz")

    if not op.exists(bm_auto):
        img = nb.load(t1)
        mask = nb.load(skullstripped)
        bmask = new_img_like(mask, np.asanyarray(mask.dataobj) > 0)
        resampled_mask = resample_to_img(bmask, img, "nearest")
        masked_image = new_img_like(
            img, np.asanyarray(img.dataobj) * resampled_mask.dataobj
        )
        masked_image.to_filename(bm_auto)

    if not op.exists(bm):
        copyfile(bm_auto, bm, copy=True, use_hardlink=True)

    return subjects_dir, subject_id

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~??????

@mark.task
@mark.annotate({"return": {"": }})
def MakeMidthickness(graymid:Path): #InputMultiPath as Path?
    """
    Variation on MRIsExpand that checks for an existing midthickness/graymid surface.

    ``mris_expand`` is an expensive operation, so this avoids re-running it when the
    working directory is lost.
    If users provide their own midthickness/graymid file, we assume they have
    created it correctly.

    """
    @property #?
    def cmdline():
        cmd = super(MakeMidthickness).cmdline
        if not isdefined(graymid) or len(graymid) < 1:
            return cmd

        # Possible graymid values inclue {l,r}h.{graymid,midthickness}
        # Prefer midthickness to graymid, require to be of the same hemisphere
        # as input
        source = None
        in_base = Path(graymid).name #in_base = Path(self.inputs.in_file).name
        mt = graymid._associated_file(in_base, "midthickness")
        gm = graymid._associated_file(in_base, "graymid")

        for surf in graymid:
            if Path(surf).name == mt:
                source = surf
                break
            if Path(surf).name == gm:
                source = surf

        if source is None:
            return cmd

        return "cp {} {}".format(source, self._list_outputs()["out_file"])#???

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?????

@mark.task
@mark.annotate({"return": {"lta_outputs": }})
def PatchedLTAConvert(TruncateLTA, LTAConvert): #inputs: 2classes
    lta_outputs = ("out_lta",)
    return lta_outputs

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~???????

@mark.task
@mark.annotate({"return": {"lta_outputs": }})
def PatchedRobustRegister(TruncateLTA, RobustRegister): #inputs: 2classes
    lta_outputs = ("out_reg_file", "half_source_xfm", "half_targ_xfm")
    return lta_outputs

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ok#?newpath=runtime.cwd ==> newpath=os.getcwd()

@mark.task
@mark.annotate({"return": {"out_file": File}})
def RefineBrainMask(in_anat: File, in_aseg: File, in_ants: File): 
    import os
    out_file = fname_presuffix(
        in_anat, suffix="_rbrainmask", newpath=os.getcwd() 
    ) #?newpath=runtime.cwd ==> newpath=os.getcwd()

    anatnii = nb.load(in_anat)
    msknii = nb.Nifti1Image(
        grow_mask(
            anatnii.get_fdata(dtype="float32"),
            np.asanyarray(nb.load(in_aseg).dataobj).astype("int16"),
            np.asanyarray(nb.load(in_ants).dataobj).astype("int16"),
        ),
        anatnii.affine,
        anatnii.header,
    )
    msknii.set_data_dtype(np.uint8)
    msknii.to_filename(out_file)

    return out_file


def grow_mask(anat, aseg, ants_segs=None, ww=7, zval=2.0, bw=4):
    """
    Grow mask including pixels that have a high likelihood.

    GM tissue parameters are sampled in image patches of ``ww`` size.
    This is inspired on mindboggle's solution to the problem:
    https://github.com/nipy/mindboggle/blob/master/mindboggle/guts/segment.py#L1660

    """
    from skimage import morphology as sim

    selem = sim.ball(bw)

    if ants_segs is None:
        ants_segs = np.zeros_like(aseg, dtype=np.uint8)

    aseg[aseg == 42] = 3  # Collapse both hemispheres
    gm = anat.copy()
    gm[aseg != 3] = 0

    refined = refine_aseg(aseg)
    newrefmask = sim.binary_dilation(refined, selem) - refined
    indices = np.argwhere(newrefmask > 0)
    for pixel in indices:
        # When ATROPOS identified the pixel as GM, set and carry on
        if ants_segs[tuple(pixel)] == 2:
            refined[tuple(pixel)] = 1
            continue

        window = gm[
            pixel[0] - ww:pixel[0] + ww,
            pixel[1] - ww:pixel[1] + ww,
            pixel[2] - ww:pixel[2] + ww,
        ]
        if np.any(window > 0):
            mu = window[window > 0].mean()
            sigma = max(window[window > 0].std(), 1.0e-5)
            zstat = abs(anat[tuple(pixel)] - mu) / sigma
            refined[tuple(pixel)] = int(zstat < zval)

    refined = sim.binary_opening(refined, selem)
    return refined


def refine_aseg(aseg, ball_size=4):
    """
    Refine the ``aseg.mgz`` mask of Freesurfer.

    First step to reconcile ANTs' and FreeSurfer's brain masks.
    Here, the ``aseg.mgz`` mask from FreeSurfer is refined in two
    steps, using binary morphological operations:

      1. With a binary closing operation the sulci are included
         into the mask. This results in a smoother brain mask
         that does not exclude deep, wide sulci.

      2. Fill any holes (typically, there could be a hole next to
         the pineal gland and the corpora quadrigemina if the great
         cerebral brain is segmented out).

    """
    from skimage import morphology as sim
    from scipy.ndimage.morphology import binary_fill_holes

    # Read aseg data
    bmask = aseg.copy()
    bmask[bmask > 0] = 1
    bmask = bmask.astype(np.uint8)

    # Morphological operations
    selem = sim.ball(ball_size)
    newmask = sim.binary_closing(bmask, selem)
    newmask = binary_fill_holes(newmask.astype(np.uint8), selem).astype(np.uint8)

    return newmask.astype(np.uint8)


