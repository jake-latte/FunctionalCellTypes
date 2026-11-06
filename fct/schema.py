"""Building blocks of every config model: the strict base model and the constrained types.

They live apart from `fct.config` so that modules whose own config is a section of `Config`
(`fct.spatial`) can use them without importing `fct.config`, which imports those sections.
"""

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

# Constrained types: the config is the boundary, so impossible values fail at load time.
PosInt = Annotated[int, Field(ge=1)]
NonNegInt = Annotated[int, Field(ge=0)]
PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def exclude_at_default(model: BaseModel, table: Mapping[str, Any]) -> dict[str, Any]:
    """A nested pydantic `exclude` for every dotted path of `table` at which `model` holds the
    value listed for it. This is how a field added after artefacts were written stays out of
    their hashes and texts: at that value it is left out, so everything that does not set it
    is byte-identical to what it was before the field existed. A path whose parent is None
    (an unset optional section) is skipped; an unknown field raises AttributeError."""
    exclude: dict[str, Any] = {}
    for path, value in table.items():
        *parents, leaf = path.split(".")
        node, target = model, exclude
        for name in parents:
            node = getattr(node, name)
            if node is None:
                break
            target = target.setdefault(name, {})
        else:
            if getattr(node, leaf) == value:
                target[leaf] = True
    return {name: sub for name, sub in exclude.items() if sub != {}}
