"""
ARIA — Autonomous Reasoning and Intelligence Architecture
Version 1.1.0

Core exports:
"""
from .agents.base import BaseAgent, aria_agent
from .core.improvement import CapabilityGap, Lesson, SelfImprovementEngine
from .core.models import ARIAConfig, ExecutionContract, MemoryRing, TaskResult, TaskStatus
from .core.orchestrator import ARIAOrchestrator
from .core.self_aware.consciousness import SelfConsciousness
from .permissions.engine import Capability, PermissionDeniedError, PermissionEngine
from .security.patcher import SelfPatcher
from .security.scanner import ScanReport, SecurityScanner

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
