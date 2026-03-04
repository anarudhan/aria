"""
ARIA — Autonomous Reasoning and Intelligence Architecture
Version 1.1.0

Core exports:
"""
from .core.orchestrator import ARIAOrchestrator
from .core.models import ARIAConfig, ExecutionContract, TaskResult, TaskStatus, MemoryRing
from .agents.base import BaseAgent, aria_agent
from .permissions.engine import PermissionEngine, Capability, PermissionDeniedError
from .security.scanner import SecurityScanner, ScanReport
from .security.patcher import SelfPatcher
from .core.self_aware.consciousness import SelfConsciousness
from .core.improvement import SelfImprovementEngine, Lesson, CapabilityGap

__version__ = "1.1.0"
__all__ = [
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
