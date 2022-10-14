"""niworkflows.interfaces.norm.SpatialNormalization()"""



import typing as ty

from pydra import mark
from pydra.core.specs import File

import numpy as np
import nibabel as nb
import os

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?ALLLLL~~~~~~~~~~multiple funcs: only _run_interface()?
@mark.task
@mark.annotate({
    "return": {
        "reference_image": File
    }
})
def SpatialNormalization(package_version: float, moving_image: File, reference_image: File, moving_mask: File, reference_mask: File, lesion_mask: File, num_threads: int, falvor: str, orientation: str, reference: str, moving: str, template: str, settings: ty.List(), template_spec: ty.Any, template_resolution: int, explicit_masking: bool, initial_moving_transform: File, float: bool):
# Get a list of settings files.
    settings_files = self._get_settings()
    ants_args = self._get_ants_args()

    if not isdefined(initial_moving_transform):
        NIWORKFLOWS_LOG.info("Estimating initial transform using AffineInitializer")
        init = AffineInitializer(
            fixed_image=ants_args["fixed_image"],
            moving_image=ants_args["moving_image"],
            num_threads=num_threads,
        )
        init.resource_monitor = False
        init.terminal_output = "allatonce"
        init_result = init.run()
        # Save outputs (if available)
        init_out = _write_outputs(init_result.runtime, ".nipype-init")
        if init_out:
            NIWORKFLOWS_LOG.info(
                "Terminal outputs of initialization saved (%s).",
                ", ".join(init_out),
            )

        ants_args["initial_moving_transform"] = init_result.outputs.out_file

    # For each settings file...
    for ants_settings in settings_files:

        NIWORKFLOWS_LOG.info("Loading settings from file %s.", ants_settings)
        # Configure an ANTs run based on these settings.
        self.norm = Registration(from_file=ants_settings, **ants_args)
        self.norm.resource_monitor = False
        self.norm.terminal_output = self.terminal_output

        cmd = self.norm.cmdline
        # Print the retry number and command line call to the log.
        NIWORKFLOWS_LOG.info("Retry #%d, commandline: \n%s", self.retry, cmd)
        self.norm.ignore_exception = True
        with open("command.txt", "w") as cmdfile:
            print(cmd + "\n", file=cmdfile)

        # Try running registration.
        interface_result = self.norm.run()

        if interface_result.runtime.returncode != 0:
            NIWORKFLOWS_LOG.warning("Retry #%d failed.", self.retry)
            # Save outputs (if available)
            term_out = _write_outputs(
                interface_result.runtime, ".nipype-%04d" % self.retry
            )
            if term_out:
                NIWORKFLOWS_LOG.warning(
                    "Log of failed retry saved (%s).", ", ".join(term_out)
                )
        else:
            runtime.returncode = 0
            # Note this in the log.
            NIWORKFLOWS_LOG.info(
                "Successful spatial normalization (retry #%d).", self.retry
            )
            # Break out of the retry loop.
            return runtime

        self.retry += 1

    # If all tries fail, raise an error.
    raise RuntimeError(
        "Robust spatial normalization failed after %d retries." % (self.retry - 1)
    )