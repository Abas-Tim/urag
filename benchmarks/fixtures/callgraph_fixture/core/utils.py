import os.path as opath


def is_file(path):
    return opath.exists(path)


def is_dir(path):
    return opath.isdir(path)
