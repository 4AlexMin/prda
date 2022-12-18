""" The :mod:`prda.graphic` contains plot methods and color assignment.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyecharts.options as opts
import seaborn as sns
from pyecharts.charts import Line, Scatter3D

__all__ = ['assign_colors', 'lineplot_html', 'scatter_3d_html', 'heatmap']

def __make_path(fpath):
    if '/' in fpath:
        path_to_make = ''
        for folder in fpath.split('/')[:-1]:
            path_to_make += folder + '/'
        os.makedirs(path_to_make, exist_ok=True)

def __process_path(title: str, filepath: str, suffix='png')-> None:
    if not filepath:
        filepath =  title + '.' + suffix
    else:
        __make_path(filepath)
    return filepath


def __random_color():
    color_arr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    rgb = "#"
    for i in range(6):
        rgb += color_arr[random.randint(0,14)]
    return rgb

def assign_colors(x, colormap=None, return_colormap=False):
    """
    Map discrete sequence from discrete values to colors. If `np.nan` contains, treated as a seperate class.
    
    This function will not change `x`

    Parameters
    ----------
    x : array_like
        Discrete sequence to be mapped.
    """
    y = x.copy()
    if colormap is None:
        val_uniques = set()
        val_uniques.update(y)
        colormap = {val: __random_color() for val in val_uniques}
    
    # map sequence using `colormap`
    if type(y) == pd.Series:
        if y.hasnans:
            nan_color = '#EFEFEF'
            y.fillna(nan_color, inplace=True)
            colormap[nan_color] = nan_color
            colormap[np.nan] = nan_color
            # y.fillna(color_map[np.nan], inplace=True) #KeyError
        y = y.apply(lambda val: colormap[val]) # Now y doesnot contains np.nan
        colormap.pop(nan_color, None)
    else:
        y = [colormap[val] for val in y]
    
    if return_colormap:
        return y, colormap
    else:
        return y


def lineplot_html(data: pd.DataFrame, x: str = None, y: list = None, markpoints: dict = None, title='lineplot', filepath=None):
    """
    Draw lineplot using columns in `data`

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    x : str, optional
        column to be used as x, if x=`None` then use index_col as default. by default None
    y : list[str], optional
        column to be used as y, if y=`None` then use every columns in `data`. by default None
    title : str, optional
        _description_, by default 'lineplot'
    filepath : _type_, optional
        _description_, by default None
    """

    filepath = __process_path(title, filepath, suffix='html')
    
    # Get data for x,y axis.
    if x == None:
        x_data = [str(_) for _ in data.index.to_list()]
    else:
        x_data = [str(_) for _ in data.loc[:, x]]
    y_datas = dict()
    if y == None:
        for col in data.columns:
            y_datas[col] = [round(_, 3) for _ in data.loc[:, col]]
    else:
        for col in y:
            y_datas[col] = [round(_, 3) for _ in data.loc[:, col]]
    
    # Draw lineplot here
    lineplot = Line(init_opts=opts.InitOpts(width="1600px", height="800px"))
    lineplot.add_xaxis(xaxis_data=x_data)
    for k, v in y_datas.items():
        if markpoints:
            if k in markpoints.keys():
                markpoint_items = [opts.MarkPointItem(coord=[str(point[0]), point[1]]) for point in markpoints[k]]
            else:
                markpoint_items = []
            
            lineplot.add_yaxis(
                series_name=k,
                y_axis=v,
                markpoint_opts=opts.MarkPointOpts(
                data=markpoint_items, symbol='triangle', symbol_size=10),
                markline_opts=opts.MarkLineOpts(
                    data=[opts.MarkLineItem(type_="average", name=k+'_avg')]))
        else:
            lineplot.add_yaxis(
                series_name=k,
                y_axis=v,
                # markpoint_opts=opts.MarkPointOpts(
                #     data=[
                #         opts.MarkPointItem(type_="max", name="最大值"),
                #         opts.MarkPointItem(type_="min", name="最小值"),
                #     ]
                # ),
                markline_opts=opts.MarkLineOpts(
                    data=[opts.MarkLineItem(type_="average", name=k+'_avg')]))

    lineplot.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        datazoom_opts=opts.DataZoomOpts(is_show=True),
        # xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    lineplot.render(filepath)


def scatter_3d_html(data: pd.DataFrame, x: str, y: str, z: str, color_hue: str = None, size_hue: str = None, title: str = 'scatter_3d', filepath: str = None):
    """Draw a 3d scatter plot using assigned columns in `data`.
    
    Output plot is in the form of filetype 'html'.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    x : str
        column name of `data` to draw as x-axis
    y : str
        column name of `data` to draw as y-axis
    z : str
        column name of `data` to draw as z-axis
    color_hue : str, optional
        visual map indicator of scatters' color, also is a column name of `data`, by default None
    size_hue : str, optional
        scatter size indicator,also a column name of `data`, by default None
    title : str, optional
        _description_, by default 'scatter_3d'
    filepath : str, optional
        _description_, by default None

    References
    ----------
    [1] docsify, pyecharts-gallery Scatter3d, https://gallery.pyecharts.org/#/Scatter3D/scatter3d
    """
    filepath = __process_path(title, filepath, suffix='html')

    # deal with data first
    use_cols = [col for col in [x, y, z, color_hue, size_hue] if col]
    vals = data.loc[:, use_cols].values.astype(np.dtype('float32')).tolist()

    # plot second
    scatter =  Scatter3D(init_opts=opts.InitOpts(width="1440px", height="720px"))  # bg_color="black"
    scatter.add(
        series_name="",
        data=vals,
        xaxis3d_opts=opts.Axis3DOpts(
            name=x,
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        yaxis3d_opts=opts.Axis3DOpts(
            name=y,
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        zaxis3d_opts=opts.Axis3DOpts(
            name=z,
            type_="value",
            # textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        grid3d_opts=opts.Grid3DOpts(width=100, height=100, depth=100),
    )
    vmap_opts = []
    if color_hue:
        color_max = data.loc[:, color_hue].max()
        vmap_opts.append(
            opts.VisualMapOpts(
                type_="color",
                is_calculable=True,
                dimension=3,
                pos_top="10",
                max_=color_max,
                range_color=[
                    "#1710c0",
                    "#0b9df0",
                    "#00fea8",
                    "#00ff0d",
                    "#f5f811",
                    "#f09a09",
                    "#fe0300",
                ],
            )
        )
    if size_hue:
        size_max = data.loc[:, size_hue].max()
        vmap_opts.append(
            opts.VisualMapOpts(
                type_="size",
                is_calculable=True,
                dimension=4,
                pos_bottom="10",
                max_=size_max,
                range_size=[10, 40],
            )
        )
    scatter.set_global_opts(visualmap_opts=vmap_opts)
    scatter.render(filepath)
    

def heatmap(data: pd.DataFrame, title: str = 'correlation_heatmap', filepath=None) -> None:
    """
    Currently, only support drawing correlation heatmap along `data` columns.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    title : str
        title of figure
    """
    filepath = __process_path(title, filepath)

    data_corr = np.corrcoef(data, rowvar = False)

    sns.set(style="white")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(data_corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(25, 15))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio 
    sns.heatmap(data_corr, cmap=cmap,  center=0,mask = mask,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels = data.columns, yticklabels = data.columns)

    ax.set_title(title)
    plt.savefig(filepath, dpi = 800,bbox_inches = 'tight')


