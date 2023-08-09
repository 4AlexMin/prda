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
from lifelines import KaplanMeierFitter

__all__ = ['random_color', 'assign_colors', 'lineplot_html', 'scatter_3d_html', 'heatmap', 'survival_plot', 'box_whisker_plot']

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

def __handle_axes(ax, log_: bool = False):
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(12,8))
    if log_:
        ax.set_xscale("log")
    return ax

def random_color():
    color_arr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    rgb = "#"
    for i in range(6):
        rgb += color_arr[random.randint(0,14)]
    return rgb

def __format_digit(x):
    """Formatting a single digit. \
        This is necessary as "pyechats" sometimes generates unexpected output if number is not in the right format.

    Parameters
    ----------
    x : interger, real or str
        single value to be formatted.
    """
    if type(x) == str:
        return x
    else:
        return round(np.float64(x), 5)



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
        colormap = {val: random_color() for val in val_uniques}
    
    # map sequence using `colormap`
    if type(y) == pd.Series:
        
        # Manually assign a color for nan
        if y.hasnans:
            nan_color = '#EFEFEF'
            y.fillna(nan_color, inplace=True)  # Now y doesnot contains np.nan
            colormap[nan_color] = nan_color

            y = y.apply(lambda val: colormap[val])
            colormap.pop(nan_color, None)

        else:
            y = y.apply(lambda val: colormap[val])
    else:
        y = [colormap[val] for val in y]
    
    if return_colormap:
        return y, colormap
    else:
        return y


def lineplot_html(data: pd.DataFrame, x: str = None, y: list = None, markpoints: dict = None, title='lineplot', filepath=None) -> None:
    """Draw lineplot using columns in `data`

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    x : str, optional
        column to be used as x, if x=`None` then use index_col as default. by default None
    y : list[str], optional
        column to be used as y, if y=`None` then use every columns in `data`. by default None
    markpoints : dict, optional
        emphasize points in several specific lines, in the form of `{column: points}` by default None
    title : str, optional
        figure title, by default 'lineplot'
    filepath : _type_, optional
        _description_, by default None
    """


    filepath = __process_path(title, filepath, suffix='html')
    
    # Get data for x,y axis.
    if x == None:
        x_data = [str(__format_digit(_)) for _ in data.index.to_list()]
    else:
        x_data = [str(__format_digit(_)) for _ in data.loc[:, x]]
    y_datas = dict()
    if y == None:
        for col in data.columns:
            y_datas[col] = [__format_digit(_) for _ in data.loc[:, col]]
    else:
        for col in y:
            y_datas[col] = [__format_digit(_) for _ in data.loc[:, col]]
    
    # Draw lineplot here
    lineplot = Line(init_opts=opts.InitOpts(width="1600px", height="800px"))
    lineplot.add_xaxis(xaxis_data=x_data)
    for k, v in y_datas.items():
        if markpoints:
            if k in markpoints.keys():
                markpoint_items = [opts.MarkPointItem(coord=[__format_digit(point[0]), __format_digit(point[1])]) for point in markpoints[k]]
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
        title within the plot, by default 'scatter_3d'
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
    

def heatmap(data: pd.DataFrame, annot: bool = True, ax=None, title: str = 'correlation_heatmap', filepath=None) -> None:
    """
    Currently, only support drawing correlation heatmap along `data` columns.

    Parameters
    ----------
    data : pd.DataFrame
        The input data.
    annot : bool
        If True, write the data value in each cell.\
        Note that DataFrames will match on position, not index, by default True.
    title : str
        Title of figure.
    """
    filepath = __process_path(title, filepath)

    data_corr = np.corrcoef(data, rowvar = False)

    sns.set(style="white")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(data_corr, dtype=np.bool))

    # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(25, 15))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = __handle_axes(ax)
    sns.heatmap(
        data_corr, 
        cmap=cmap, 
        center=0, 
        mask = mask, 
        annot=annot,       
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5}, 
        xticklabels = data.columns, 
        yticklabels = data.columns,
        ax=ax
        )
    plt.title(title)
    plt.savefig(filepath, dpi=600, bbox_inches = 'tight')


def survival_plot(data, duration: str, status: str, hue: str, alpha: float = 0.05, ax=None, filepath='survial_analysis_plot.png'):
    """Plot Kaplan-Meier estimate for the survival analysis.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    duration : str
        column name for survial duration.
    status : str
        column name indicating survial status.
    hue : str
        column name for individual types.
    alpha : float, optional
        The alpha value associated with the confidence intervals, by default 0.05.
    ax : matplotlib Axes, optional
        _description_, by default None
    filepath : str, optional
        _description_, by default 'survial_analysis_plot.png'
    """
    ax = __handle_axes(ax)
    ax.set_xlabel('Survival Time', fontdict={'size': 16})
    ax.set_ylabel('Overall Survival', fontdict={'size': 16})
    
    # Kaplan-Meier estimation
    kmf = KaplanMeierFitter(alpha=alpha)
    for type in data.loc[:, hue].unique():
        popul = data.loc[data.HE == type]
        kmf.fit(durations=popul.loc[:, duration], event_observed=popul.loc[:, status], label=type)
        kmf.plot(ax=ax)

    plt.savefig(filepath, dpi=600)


def box_whisker_plot(data: pd.DataFrame, usecols=None, whis: float = 1.5, observations: bool = True, ax=None, filepath='box_and_whisker_plot.png'):
    """Draw box plot (or box-and-whisker plot) shows the distribution of quantitative data.

    Parameters
    ----------
    data : pd.DataFrame
        Treated as long-form data.
    usecols : _type_, optional
        the columns to draw as y-axis, if `None`, use all columns, by default None
    whis : float, optional
        Maximum length of the plot whiskers as `proportion` of the interquartile range. \
        Whiskers extend to the furthest datapoint within that range. More extreme points are marked as outliers, by default 1.5
    observations : bool, optional
        Whether draw observations, by default True
    ax : matplotlib Axes, optional
        _description_, by default None
    filepath : str, optional
        _description_, by default 'box_and_whisker_plot.png'
    
    References
    ----------
    [1] Horizontal boxplot with observations — seaborn 0.12.2 documentation. https://seaborn.pydata.org/examples/horizontal_boxplot.html.

    """
    sns.set_theme(style="ticks")
    ax = __handle_axes(ax)
    if not usecols:
        view = data
    else:
        view = data.loc[:, usecols]

    # Plot the orbital period with horizontal boxes
    sns.boxplot(data=view, whis=whis, fliersize=0.5, width=.6, palette="vlag", orient='h')

    # Add in points to show each observation
    if observations:
        sns.stripplot(data=view, size=.5, linewidth=0, palette="gray", orient='h')

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    sns.despine(trim=True, left=True)

    plt.savefig(filepath, dpi=600)
