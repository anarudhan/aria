from .improvement import CapabilityGap, Lesson, SelfImprovementEngine
from .models import (
    ARIAConfig,
    CapabilityFingerprint,
    ExecutionContract,
    ExecutionMode,
    MemoryEntry,
    MemoryRing,
    TaskGenome,
    TaskNode,
    TaskResult,
    TaskStatus,
    TrustRecord,
)


def __getattr__(name: str):
    if name == "ARIAOrchestrator":
        from .orchestrator import ARIAOrchestrator
        return ARIAOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARIAConfig",
    "ARIAOrchestrator",
    "CapabilityFingerprint",
    "CapabilityGap",
    "ExecutionContract",
    "ExecutionMode",
    "Lesson",
    "MemoryEntry",
    "MemoryRing",
    "SelfImprovementEngine",
    "TaskGenome",
    "TaskNode",
    "TaskResult",
    "TaskStatus",
    "TrustRecord",
]
