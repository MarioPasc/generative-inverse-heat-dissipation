"""Exception hierarchy of :mod:`ihdm.train` (``02-engineering-practices.md`` §2)."""


class TrainError(Exception):
    """Raised when a training artefact cannot be written or read back consistently."""
