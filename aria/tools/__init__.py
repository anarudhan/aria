"""
ARIA tools: the declaration layer of the guarded tool pipeline (issue #42).

This package only declares and validates. Nothing here enforces a permission
yet; see docs/tools.md.
"""

from .decorator import SPEC_ATTRIBUTE, tool
from .freeze import FreezeError, FrozenArgs, FrozenMap, freeze_args
from .registry import ToolRegistry
from .spec import RISK_ORDER, TOOL_NAME_PATTERN, Need, ToolSpec, ToolSpecError

__all__ = [
    "RISK_ORDER",
    "SPEC_ATTRIBUTE",
    "TOOL_NAME_PATTERN",
    "FreezeError",
    "FrozenArgs",
    "FrozenMap",
    "Need",
    "ToolRegistry",
    "ToolSpec",
    "ToolSpecError",
    "freeze_args",
    "tool",
]
