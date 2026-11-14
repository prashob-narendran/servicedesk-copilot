"""One error type for the whole service, rendered as {"error": {code, message, request_id}}."""


class ServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class ModelNotReady(ServiceError):
    def __init__(self, component: str, detail: str = ""):
        msg = f"{component} is not available"
        if detail:
            msg += f": {detail}"
        super().__init__(503, "MODEL_NOT_READY", msg)


class UpstreamError(ServiceError):
    """Failures of the external LLM API."""
