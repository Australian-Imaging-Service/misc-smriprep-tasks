"""niworkflows.interfaces.surf.Path2BIDS()"""

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~?it has more than one function inside the class... what should I do?
@mark.task
@mark.annotate({"return": {"extension":str}}) 
def Path2BIDS(in_file:File):
    import re
    _pattern = re.compile(
        r"(?P<hemi>[lr])h.(?P<suffix>(wm|smoothwm|pial|midthickness|"
        r"inflated|vinflated|sphere|flat))[\w\d_-]*(?P<extprefix>\.\w+)?"
    )
    _excluded = ("extprefix",)

    
    in_file = Path(in_file)
    extension = "".join(in_file.suffixes[-((in_file.suffixes[-1] == ".gz") + 1):])
    info = _pattern.match(in_file.name[: -len(extension)]).groupdict() #???
    extension = f"{info.pop('extprefix', None) or ''}{extension}"
    self._results.update(info) #self._results.update(info) #?????????
    if "hemi" in self._results: #?????
        hemi = hemi.upper()
    return extension #?

   