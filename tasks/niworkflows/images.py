"""niworkflows.interfaces.images.Conform(),
niworkflows.interfaces.images.TemplateDimensions()"""



import typing as ty

from pydra import mark
from pydra.core.specs import File

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?func and sub-funcs
@mark.task
@mark.annotate({
    "return": {
        "t1w_valid_list": ty.List[File],
        "target_zooms": ty.Tuple[float, float, float],
        "target_shape": ty.Tuple[int, int, int],
        "out_report": File
    }
})
def template_dimensions(t1w_list: ty.List[File], max_scale: float):
    import numpy as np
    import nibabel as nb
    import os
    # Load images, orient as RAS, collect shape and zoom data
    in_names = np.array(t1w_list)
    orig_imgs = np.vectorize(nb.load)(in_names)
    reoriented = np.vectorize(nb.as_closest_canonical)(orig_imgs)
    all_zooms = np.array([img.header.get_zooms()[:3] for img in reoriented])
    all_shapes = np.array([img.shape[:3] for img in reoriented])

    # Identify images that would require excessive up-sampling
    valid = np.ones(all_zooms.shape[0], dtype=bool)
    while valid.any():
        target_zooms = all_zooms[valid].min(axis=0)
        scales = all_zooms[valid] / target_zooms
        if np.all(scales < max_scale):
            break
        valid[valid] ^= np.any(scales == scales.max(), axis=1)

    # Ignore dropped images
    valid_fnames = np.atleast_1d(in_names[valid]).tolist()
    t1w_valid_list = valid_fnames

    # Set target shape information
    target_zooms = all_zooms[valid].min(axis=0)
    target_shape = all_shapes[valid].max(axis=0)

    target_zooms = tuple(target_zooms.tolist())
    target_shape = tuple(target_shape.tolist())

    # Create report
    dropped_images = in_names[~valid]
    segment = _generate_segment(dropped_images, target_shape, target_zooms) #?
    out_report = os.path.join(os.getcwd(), "report.html") #?runtime.cwd
    with open(out_report, "w") as fobj:
        fobj.write(segment)

    out_report = out_report

    return t1w_valid_list, target_zooms, target_shape, out_report


def _generate_segment(discards, dims, zooms):
   import os
   task = template_dimensions() #?
   t1w_list = task.inputs.t1w_list #?

   items = [
        DISCARD_TEMPLATE.format(path=path, basename=os.path.basename(path))
        for path in discards
    ]
    discard_list = (
        "\n".join(["\t\t\t<ul>"] + items + ["\t\t\t</ul>"]) if items else ""
    )
    zoom_fmt = "{:.02g}mm x {:.02g}mm x {:.02g}mm".format(*zooms)
    return CONFORMATION_TEMPLATE.format(
        n_t1w=len(t1w_list),
        dims="x".join(map(str, dims)),
        zooms=zoom_fmt,
        n_discards=len(discards),
        discard_list=discard_list,
    )


CONFORMATION_TEMPLATE = """\t\t<h3 class="elem-title">Anatomical Conformation</h3>
\t\t<ul class="elem-desc">
\t\t\t<li>Input T1w images: {n_t1w}</li>
\t\t\t<li>Output orientation: RAS</li>
\t\t\t<li>Output dimensions: {dims}</li>
\t\t\t<li>Output voxel size: {zooms}</li>
\t\t\t<li>Discarded images: {n_discards}</li>
{discard_list}
\t\t</ul>
"""
DISCARD_TEMPLATE = """\t\t\t\t<li><abbr title="{path}">{basename}</abbr></li>"""


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@mark.task
@mark.annotate({
    "return": {
        "out_file": File,
        "transform": File
    }
})
def Conform(in_file: File, target_zooms: ty.Tuple[float, float, float], target_shape: ty.Tuple[int, int, int]):
    import nibabel as nb
    import numpy as np
    import os
    from nipype.utils.filemanip import fname_presuffix #??
    
    # Load image, orient as RAS
    fname = in_file
    orig_img = nb.load(fname)
    reoriented = nb.as_closest_canonical(orig_img)

    # Set target shape information
    target_zooms = np.array(target_zooms)
    target_shape = np.array(target_shape)
    target_span = target_shape * target_zooms

    zooms = np.array(reoriented.header.get_zooms()[:3])
    shape = np.array(reoriented.shape[:3])

    # Reconstruct transform from orig to reoriented image
    ornt_xfm = nb.orientations.inv_ornt_aff(
        nb.io_orientation(orig_img.affine), orig_img.shape
    )
    # Identity unless proven otherwise
    target_affine = reoriented.affine.copy()
    conform_xfm = np.eye(4)

    xyz_unit = reoriented.header.get_xyzt_units()[0]
    if xyz_unit == "unknown":
        # Common assumption; if we're wrong, unlikely to be the only thing that breaks
        xyz_unit = "mm"

    # Set a 0.05mm threshold to performing rescaling
    atol_gross = {"meter": 5e-5, "mm": 0.05, "micron": 50}[xyz_unit]
    # if 0.01 > difference > 0.001mm, freesurfer won't be able to merge the images
    atol_fine = {"meter": 1e-6, "mm": 0.001, "micron": 1}[xyz_unit]

    # Update zooms => Modify affine
    # Rescale => Resample to resized voxels
    # Resize => Resample to new image dimensions
    update_zooms = not np.allclose(zooms, target_zooms, atol=atol_fine, rtol=0)
    rescale = not np.allclose(zooms, target_zooms, atol=atol_gross, rtol=0)
    resize = not np.all(shape == target_shape)
    resample = rescale or resize
    if resample or update_zooms:
        # Use an affine with the corrected zooms, whether or not we resample
        if update_zooms:
            scale_factor = target_zooms / zooms
            target_affine[:3, :3] = reoriented.affine[:3, :3] @ np.diag(
                scale_factor
            )

        if resize:
            # The shift is applied after scaling.
            # Use a proportional shift to maintain relative position in dataset
            size_factor = target_span / (zooms * shape)
            # Use integer shifts to avoid unnecessary interpolation
            offset = (
                reoriented.affine[:3, 3] * size_factor - reoriented.affine[:3, 3]
            )
            target_affine[:3, 3] = reoriented.affine[:3, 3] + offset.astype(int)

        conform_xfm = np.linalg.inv(reoriented.affine) @ target_affine

        # Create new image
        data = reoriented.dataobj
        if resample:
            import nilearn.image as nli

            data = nli.resample_img(reoriented, target_affine, target_shape).dataobj
        reoriented = reoriented.__class__(data, target_affine, reoriented.header)

    # Image may be reoriented, rescaled, and/or resized
    if reoriented is not orig_img:
        out_name = fname_presuffix(fname, suffix="_ras", newpath=os.getcwd())#newpath=runtime.cwd
        reoriented.to_filename(out_name)
    else:
        out_name = fname

    transform = ornt_xfm.dot(conform_xfm)
    if not np.allclose(orig_img.affine.dot(transform), target_affine):
        raise ValueError("Original and target affines are not similar")

    mat_name = fname_presuffix(
        fname, suffix=".mat", newpath=os.getcwd(), use_ext=False
    )#newpath=runtime.cwd
    np.savetxt(mat_name, transform, fmt="%.08f")

    out_file = out_name
    transform = mat_name

    return out_file, transform