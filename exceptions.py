class PiiMaskerError(Exception):
    pass


class ConfigValidationError(PiiMaskerError):
    pass


class LayerInitError(PiiMaskerError):
    pass


class LlmApiError(PiiMaskerError):
    pass


class MaskingError(PiiMaskerError):
    pass
