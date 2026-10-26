"""Stable seeds. Python's hash() of strings is randomised per process, so we
never use it for seeding: reruns must give identical numbers."""
import zlib


def stable_seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode()) & 0xFFFFFFFF
