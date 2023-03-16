from .mhcan import MHCAN


def build_model(args):
    return MHCAN(args)
