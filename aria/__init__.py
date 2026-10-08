"""
ARIA — Autonomous Reasoning and Intelligence Architecture
Version 1.3.0

Core exports:
"""
from ._version import __version__
from .agents.base import BaseAgent, aria_agent
from .core.improvement import CapabilityGap, Lesson, SelfImprovementEngine
from .core.models import ARIAConfig, ExecutionContract, MemoryRing, TaskResult, TaskStatus
from .core.orchestrator import ARIAOrchestrator
from .core.self_aware.consciousness import SelfConsciousness
from .permissions.engine import Capability, PermissionDeniedError, PermissionEngine
from .security.patcher import SelfPatcher
from .security.scanner import ScanReport, SecurityScanner

__all__ = [
    "__version__",
    "ARIAOrchestrator",
    "ARIAConfig",
    "ExecutionContract",
    "TaskResult",
    "TaskStatus",
    "MemoryRing",
    "BaseAgent",
    "aria_agent",
    "PermissionEngine",
    "Capability",
    "PermissionDeniedError",
    "SecurityScanner",
    "ScanReport",
    "SelfPatcher",
    "SelfConsciousness",
    "SelfImprovementEngine",
    "Lesson",
    "CapabilityGap",
]
