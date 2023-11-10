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

def create_dirs(dirs, exist_ok=True):
    """Create all related directories.

    Parameters
    ----------
    dirs : list or str
        A list of relative directories to be created.
    """
    if type(dirs) == str:
        dirs = [dirs]
    
    # Split levels of directories.
    max_level = max([len(dir.split('/')) for dir in dirs])
    levels_dict = {i: set() for i in range(max_level)}
    for dir in dirs:
        if dir[-1] == '/':
            dir = dir[: -1]
        splited = dir.split('/')
        if '.' in splited[-1]:
            if len(splited) > 1:
                splited = splited[:-1]
            else:
                continue
        else:
            if len(splited) == 1:
                levels_dict[0].add(splited[0])
                continue
        for level in range(len(splited)):
            levels_dict[level].add('/'.join(splited[:level+1]))

    for level in range(max_level):
        for dir_ in levels_dict[level]:
            os.makedirs(dir_, exist_ok=True)
    return
