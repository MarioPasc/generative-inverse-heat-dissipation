"""Exception hierarchy for the standard dataset format module."""


class DataFormatError(Exception):
    """Raised when a dataset on disk violates the standard format contract.

    Used by :mod:`ihdm.data.format` for every validation failure; never a bare
    ``assert``.
    """
