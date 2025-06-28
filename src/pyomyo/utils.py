import struct
import sys


def pack(fmt, *args):
    return struct.pack("<" + fmt, *args)


def unpack(fmt, *args):
    return struct.unpack("<" + fmt, *args)


def multichr(ords):
    if sys.version_info[0] >= 3:
        return bytes(ords)
    else:
        return "".join(map(chr, ords))


def multiord(b):
    if sys.version_info[0] >= 3:
        return list(b)
    else:
        return map(ord, b)
