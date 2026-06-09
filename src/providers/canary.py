from __future__ import annotations

from src.providers.base import ProviderContext, NemoProviderBase


class CanaryProvider(NemoProviderBase):
    """NVIDIA Canary ASR model provider using NemoProviderBase."""

    def __init__(self, context: ProviderContext):
        super().__init__(context, provider_name="canary")
