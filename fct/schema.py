"""Building blocks of every config model: the strict base model and the constrained types.

They live apart from `fct.config` so that modules whose own config is a section of `Config`
(`fct.spatial`) can use them without importing `fct.config`, which imports those sections.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# Constrained types: the config is the boundary, so impossible values fail at load time.
PosInt = Annotated[int, Field(ge=1)]
NonNegInt = Annotated[int, Field(ge=0)]
PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
