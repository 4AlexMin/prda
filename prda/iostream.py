""" The :mod:`prda.iostream` contains methodologies for input/output.
"""

import os

__all__ = ['get_files']

def get_files(relative_path='data/'):
    """
    Can only deal with 1 level sub-folder at the moment.

    Parameters
    ----------
    relative_path : str, optional
        _description_, by default 'data/'

    Returns
    -------
    _type_
        _description_
    """
    if relative_path[-1] != '/':
        relative_path += '/'
    paths = os.listdir(relative_path)
    dirs = []
    for pathname in paths:
        if 'DS_Store' not in pathname:
            if os.path.isdir(pathname):
                thisdirs = os.listdir(relative_path+pathname)
                dirs.extend([relative_path+pathname+'/'+_ for _ in thisdirs if 'DS_Store' not in _])
            else:
                dirs.append(relative_path+pathname)

    return dirs