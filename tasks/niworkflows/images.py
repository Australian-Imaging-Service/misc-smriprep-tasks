import typing as ty

from pydra import mark
from pydra.core.specs import File


@mark.task
@mark.annotate({
    "return": {
        "t1w_valid_list": ty.List[File],
        "target_zooms": ty.Tuple[float, float, float],
    }
})
def template_dimensions(t1w_list: ty.List[File], max_scale: float):
    import numpy as np
    import nibabel as nb
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
    self._results["target_shape"] = tuple(target_shape.tolist())

    # Create report
    dropped_images = in_names[~valid]
    segment = self._generate_segment(dropped_images, target_shape, target_zooms)
    out_report = os.path.join(runtime.cwd, "report.html")
    with open(out_report, "w") as fobj:
        fobj.write(segment)

    self._results["out_report"] = out_report

    return t1w_valid_list, target_zooms