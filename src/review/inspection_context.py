"""Trusted request metadata shared by family dispatch and normal record persistence."""
from contextvars import ContextVar

INSPECTION_CONTEXT=ContextVar('inspection_context',default={})
