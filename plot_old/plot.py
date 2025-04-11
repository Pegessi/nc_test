#!python.exe 
from math import ceil
from turtle import position
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import os

from typing import *
from pandas import DataFrame
from matplotlib import font_manager


"""
Style table
"""
SAVE_PREFIX = os.environ.get('SAVE_PREFIX', "./figure/")
DATA_PATH = os.environ.get('DATA_PATH', 'logs/exp-data.xlsx')
OURS = os.environ.get('OURS', "DT-Control")
COLOR=["#1F77B4", "#FE7F0E", "#2BA02D", "#c85862", "#5898D5", "#83DED8", "#FFD966", "#FF9B9B"]
MARKERS=["o", "v", "^", "x", "s"]

# 指定字体路径
font_path = 'Arial.ttf'  # 替换为实际的字体文件路径
font_prop = font_manager.FontProperties(fname=font_path)


title_style = {
    "fontfamily" : "Arial",
    "fontweight" : "bold",
    "size" : 13,
}

xlabel_style = {
    "fontfamily" : "Arial",
    "fontweight" : "bold",
    "size"       : 12,
}

ylabel_style = {
    "fontfamily" : "Arial",
    "fontweight" : "bold",
    "size"       : 12
}

xtick_style = {
    "fontfamily" : "Arial",
    "fontweight" : "bold",
    "size"       : 11,
    # "rotation"   : 90
}

ytick_style = {
    "fontfamily" : "Arial",
    "fontweight" : "bold",
    "size"       : 11
}

legend_style = {
    # 'loc' : 'upper right',
    'frameon' : False, 
    "prop": {
        'family':'Arial', 
        # 'weight':'bold', 
        'size'  : 10,
    }
}

annot_style = {
    "ha":"center",
    "va": "top", 
    "fontfamily": "Arial", 
    "fontweight": "bold",
    "fontsize": 11,
    "color": 'w',
    'zorder': 200
}

invalid_annot_style = {
    "ha":"center",
    "va": "bottom", 
    "fontfamily": "Arial", 
    "fontweight": 'bold',
    "fontsize": 8,
    "color": 'r',
    'rotation': 90,
    'zorder': 200
}

line_style = {
    ''
}

barh_style = bar_style = {
    "edgecolor" : 'k'
}

plot_style = {}


# 嵌入式字体，SC的论文必须这么搞，警钟撅烂
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def open_excel_with_pandas(fname: str, sheet: str = None, index_col=0):
    return pd.read_excel(fname, sheet_name=sheet, index_col=index_col)


def extract_label_and_array(dic: Dict, sort_with: Callable = None):
    """
    dict type input data parser
    """
    labels = []
    arrays = []
    sorter = []
    
    for v in dic.items():
        # if int(v[0][:-2]) < 10:
        sorter.append(v)
    
    if sort_with is not None:
        sorter.sort(key=sort_with)

    for p in sorter:
        labels.append(p[0]) 
        arrays.append(p[1])
        
    return labels, arrays


def extract_label_and_array(df: DataFrame):
    """
    df type input extraction
    """
    xlabel = df.index.to_numpy()
    group_tag = df.columns.to_numpy()
    arrays = [ df[col].to_numpy() for col in df ]
    return xlabel, group_tag, arrays

def extract_database_from_excel(fname: str, sheets: List[str]):
    res = None
    xmax = []
    for sht in sheets:
        xmax.append(1e9)
        df = pd.read_excel(fname, sht)
        if res is None:
            res = {}
            for col in df:
                res[col] = []
        for col in df:
            recs = df[col].loc[df[col].notna()].to_list()
            xmax[-1] = min([xmax[-1],] + flattern(recs))
            res[col].append(np.array(recs))
    print(f"min = {min(xmax)}")
    vals = []
    for v in res.values():
        add = []
        for i, ar in enumerate(v):
            # add.append(xmax[i] / ar)
            add.append(ar)
        vals.append(add)

    return list(res.keys()), vals

def varmean(ll: List[List]):
    def geo_mean_overflow(iterable):
        a = np.log(iterable)
        return np.exp(a.sum()/len(a))

    ret = []
    for l in ll:
        ret.append(geo_mean_overflow(l))
    return ret

def max_from_selected(arrays: List[np.ndarray], sel: np.ndarray = None):
    if sel is None:
        return max(flattern(arrays))

    ret = 0
    for ar in arrays:
        ret = max(ret, ar[sel].max())
    return ret

def build_xticks_large(num_x):
    x = 64
    ret = []
    for i in range(num_x):
        ret.append(f"{x}")
        x *= 2

    return ret

def build_xticks_small(num_x):
    ret = []
    for i in range(num_x):
        x = (i+1)*8
        ret.append(f"{x}")

    return ret

def build_yticks(xmin, xmax, preferred_ticks = 5):
    def _scale_up(domain, prefer):
        i=1
        while domain < prefer:
            domain *= 10
            i *= 10
        
        while domain > 10*prefer:
            domain = round(domain / 10)
            i /= 10
        return i

    domain = xmax - xmin
    _a = _scale_up(domain, preferred_ticks)
    domain *= _a
    xmin *= _a
    xmax *= _a
    if isinstance(domain, int):
        domain = domain + 0.0
    scale = round(domain, 0) # int(float(f"{domain:.1}"))
    while (domain + scale - 1e-6) // scale < preferred_ticks:
        cur_parts = (domain + scale - 1e-6) // scale
        div = False
        for i in range(preferred_ticks, 1, -1):
            if cur_parts * i <= preferred_ticks:
                scale //= i
                div = True
                break
        if not div:
            scale //= 2
        # if cur_parts * 6 == preferred_ticks:
        #     scale //= 3
        # elif cur_parts * 5 <= preferred_ticks:
        #     scale //= 5
        # else:
        #     scale //=2

    if _a == 1:
        return np.arange(xmin, (xmax+scale-1e-6)//scale*scale+1e-6, scale, dtype=np.int64)
    else:
        return np.arange(xmin, (xmax+scale-1e-6)//scale*scale+1e-6, scale) / _a


def pretty_num_label():
    pass

def custom_barwidth(group_size, x_tick, base_w):
    bar_norm = base_w / x_tick
    sp_norm = 1 - bar_norm
    bar_w = (bar_norm / group_size) * x_tick
    sp_w = (sp_norm / (group_size-1)) * x_tick if group_size > 1 else 0
    
    if group_size == 1:
        return bar_w, [0]

    b_s = []
    indices = np.arange(group_size) - (group_size-1) / 2
    for i in indices:
        b_s.append(i * (sp_w+bar_w))

    return bar_w, b_s


def flattern(lst):
    if isinstance(lst, Iterable):
        ret = []
        for subl in lst:
            ret += flattern(subl)
        return ret
    else:
        return [lst]

def append_anno_list(store: List, x, y, text: str, xlim = 1., ylim = None):
    to_rm = []
    for item in store:
        _x, _y, _ = item
        replace = False
        if abs(_x - x) < xlim:
            to_rm.append(item)
            if _x < x:
                _x -= 0.5*xlim
                x += 0.5*xlim
            else:
                _x += 0.5*xlim
                x -= 0.5*xlim
            replace = True

        if ylim is not None and abs(_y-y) < ylim:
            if not replace:
                to_rm.append(item)
            if _y < y:
                _y -= 0.5*ylim
                y += 0.5*ylim
            else:
                _y += 0.5*ylim
                y -= 0.5*ylim
            replace = True

        if replace:
            store.append((_x, _y, _))

    for rm in to_rm:
        store.remove(rm) 
    store.append((x, y, text))


def plot_rows(datalists, x_starts, figure, c):
    num_sub = len(datalists[0])
    axes = figure.subplots(len(datalists), num_sub)
    rotate = 0
    ha = "center"

    for datalist, x_start in zip(datalists, x_starts):
        base = datalists.index(datalist)
        if x_start < 64:
            build_xticks = build_xticks_small
        else:
            # rotate = 45
            # ha = "right"
            build_xticks = build_xticks_large

        for idx in range(num_sub):
            if len(datalists) == 1:
                ax = axes[idx]
            else:
                ax = axes[base][idx]
            dat = datalist[idx]
            num_bars = len(dat)
            ymax = max(dat)

            ax.bar(np.arange(num_bars), dat, 0.4, color=c, zorder=10)
            # for x, val in enumerate(dat):
            #     ax.text(x, val, f"{val:.2f}", ha="center", va="bottom", size=9, fontfamily="Arial", fontsize=10)
            ytk = build_yticks(0, ymax)
            ax.set_xticks(np.arange(num_bars), build_xticks(num_bars), rotation=rotate, ha=ha, fontfamily="Arial", fontweight="bold", fontsize=12)
            ax.set_yticks(ytk, fontfamily="Arial", fontweight="bold", fontsize=12)
            ax.set_ylim(0, ytk[-1])
            ax.grid(axis='y', color='0.8', zorder=0)
            ax.spines[:].set_zorder(20)
            # ax.set_ylim(0,ymax*1.1)
        if len(datalists) == 1: 
            axes[0].set_ylabel("Speedup", fontfamily="Arial", fontweight="bold", fontsize=12)
        else:
            axes[base][0].set_ylabel("Speedup", fontfamily="Arial", fontweight="bold", fontsize=12)


    # figure.supylabel("Speedup", fontfamily="Arial", fontweight="bold", fontsize=12, x=0.03)
    figure.supxlabel("Matrix Size (m = n)", fontfamily="Arial", fontweight="bold", fontsize=12, y=0.05)


def plot_one(ax: Axes, dats: List[np.ndarray], 
             colors: List[AnyStr] = COLOR,
             plot_name: str = 'bar', 
             markers: List[AnyStr] = MARKERS,
             invalid_val: str = None,
             invalid_text: str = None, 
             bar_config: Tuple[float, float] = (0.6, 0.6), 
             upper_bound: float = None, 
             plot_value: bool = False, 
             select: np.ndarray = None,
             anno_fn: Callable = None):
    assert len(dats) <= len(colors)
    assert markers is None or len(dats) <= len(markers)

    if select is None:
        select = np.arange(len(dats[0]))

    bar_w, x_off = custom_barwidth(len(dats), *bar_config) 
    bars = []
    annos = []

    for idx, dat in enumerate(dats):
        dat = dat[select]
        xloc = np.arange(dat.size) + (x_off[idx] if plot_name in ['bar', 'barh'] else 0.)
        if invalid_val is not None:
            _dat = []
            _xloc = []
            for i, val in enumerate(dat):
                if val != invalid_val:
                    _dat.append(val)
                    _xloc.append(xloc[i])
                else:
                    annos.append((xloc[i], val, invalid_text))
                    if plot_name in ['bar', 'barh']:
                        _dat.append(0)
                        _xloc.append(xloc[i])
            dat = _dat
            xloc = _xloc

        if plot_name in ['bar', 'barh']:
            bars.append(eval(f'ax.{plot_name}(xloc, dat, bar_w, color=colors[idx], zorder=10, **{plot_name}_style)'))
        else:
            bars.append(eval(f'ax.{plot_name}(xloc, dat, color=colors[idx], marker=markers[idx], zorder=10, **{plot_name}_style)'))
            

        if plot_value:
            for x, val in zip(xloc, dat):
                # ax.text(x, val, f"{val:.2f}", **annot_style)
                # append_anno_list(annos, x, val, val if anno_fn is None else anno_fn(val), 
                #                  xlim=select.size*bar_w*1.5)
                if plot_name in ['bar', 'plot']:
                    annos.append((x, val, val if anno_fn is None else anno_fn(val)))
                elif plot_name == 'barh':
                    annos.append((val, x, val if anno_fn is None else anno_fn(val)))
        
        if upper_bound is not None:
            for x, val in zip(xloc, dat):
                if val > upper_bound:
                    # ax.text(x, upper_bound, f"{val:.2f}", **annot_style)
                    append_anno_list(annos, x, upper_bound, val if anno_fn is None else anno_fn(val), 
                                     xlim=select.size*bar_w*1.5)
                    
    for anno in annos:
        if invalid_text is not None and anno[2] == invalid_text:
            ax.text(*anno, **invalid_annot_style)
        elif anno[2] != 'FAILED':
            ax.text(*anno, **annot_style)

    return bars


def plot_one_violin(ax: Axes, xticks: np.ndarray, dats: List[List[np.ndarray]], colors):
    """
        plot a violin plot on a subplot
         - 和柱状图不同，影线图复合图情况下x坐标可以重合
         - 影线图每根线输入一个数组，不知道是否要求长度一致 
         - 输入dict([str, data[][]])其中str对应xtick，二维data对应[横轴，bar数据]
    """
    assert len(dats) <= len(colors)
    bar_w, x_off = custom_barwidth(len(dats), 0.7, 0.6)

    ymax = ymin = 0.

    handles = []

    for i, p in enumerate(dats):
        dat = p
        parts = ax.violinplot(dat, xticks+x_off[i], showmeans=True, showmedians=False, widths=bar_w)
        
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.5)

        for key, pc in parts.items():
            if key =='bodies':
                continue
            pc.set_color(colors[i])
            pc.set_clim((0.1,0.1))
        
        handles.append(parts['cbars'])
        ymin = min([ymin,] + flattern(dat))
        ymax = max([ymax,] + flattern(dat))

        # tick = ((int(float(f"{ymax-ymin:.1}")) + 4) // 5 + 4) // 5 * 5
        # ymax = (ymax+tick-1e-6) // tick * tick
    return handles, ymin, ymax


def plot_one_range(ax:Axes, dats, xticks, colors, yscal, ytick_round=0, select=None):
    """
    绘制范围图
    输入: dats: Dict[str: Tuple[List, List]] 表示计算图在不同图数据上的ub，lb
    """
    assert len(dats) <= len(colors)

    bar_w, x_off = custom_barwidth(len(dats), 0.7, 0.6)

    bars = []
    legends = []

    for idx, pair in enumerate(dats.items()):
        leg, rng = pair
        if select is None:
            select = np.arange(len(rng[0]))
        ub = np.array(rng[0])[select]
        lb = np.array(rng[1])[select]
        legends.append(leg)
        xloc = np.arange(len(lb)) + x_off[idx]
        bars.append(ax.bar(xloc, ub, bar_w, lb, edgecolor='k', color=colors[idx], zorder=10))

    return bars, legends


def axestyle(ax: Axes, xticks: Iterable, yticks: Iterable, 
             xlabel: AnyStr = None, ylabel :AnyStr = None, 
             xtick_label: Union[Iterable[AnyStr], Callable] = None, ytick_label: Union[Iterable[AnyStr], Callable] =None,
             legends=None, handles=None, xlim = None, ylim = None,
             style=0):

    ax.set_xlabel(xlabel, **xlabel_style)
    ax.set_ylabel(ylabel, **ylabel_style)

    if xtick_label is None:
        xtick_label = [f"{x}" for x in xticks]
    elif isinstance(xtick_label, Callable):
        xtick_label = map(xtick_label, xticks)

    if ytick_label is None:
        ytick_label = [f"{x}" for x in yticks]
    elif isinstance(ytick_label, Callable):
        ytick_label = map(ytick_label, yticks)

    xrange = abs(max(xticks) - min(xticks))
    yrange = abs(max(yticks) - min(yticks))
    if not xlim is None:
        ax.set(xlim=(xlim[0]*xrange, xlim[1]*xrange))
    
    if not ylim is None:
        ax.set(ylim=(ylim[0]*yrange, ylim[1]*yrange))

    # special reqs
    # ax.tick_params(length=0)
    if style == 0:
        ax.grid(axis='y', color='0.8', zorder=0)
        ax.spines[:].set_linewidth(2)
        ax.spines[:].set_zorder(100)
    
    elif style == 1:
        ax.grid(axis='x', color='0.8', zorder=0)
        ax.spines[:].set_linewidth(2)
        ax.spines[:].set_zorder(100)

    elif style == 2:
        ax.spines[:].set_linewidth(0)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_zorder(100)
        ax.spines['left'].set_zorder(100)

    # if isinstance(xticks[0], float):
    #     xticks = [int(x) for x in xticks]
    # if isinstance(yticks[0], float):
    #     yticks = [int(y) for y in yticks]

    # normal reqs
    ax.set_xticks(xticks, xtick_label, **xtick_style)
    ax.set_yticks(yticks, ytick_label, **ytick_style)
    if legends is not None:
        if handles is not None:
            ax.legend(handles, legends, **legend_style)
        else:
            ax.legend(legends, **legend_style)


def grid_legend_external(ax, legends, handles):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_linewidth(0)
    ax.legend(handles,legends, **legend_style)


def figstyle(fig, labname, axes, label, label_style, is_plot_grid = False):
    if labname in ['x', 'y']:
        if isinstance(label, List):
            for ax, l in zip(axes, label):
                fn = eval(f"ax.set_{labname}label")
                fn(l, **label_style)
        elif label is not None:
            if is_plot_grid:
                fn = eval(f"fig.sup{labname}label")
                # fn(label, y=0.1, **label_style)
                fn(label, **label_style)
            else:
                fn = eval(f"axes[0].set_{labname}label")
                fn(label, **label_style)
    else: #title
        if isinstance(label, List):
            for ax, l in zip(axes, label):
                ax.set_title(l, label_style)
        elif label is not None:
            if is_plot_grid:
                fig.suptitle(label, label_style)
            else:
                axes[0].set_title(label, label_style)


# designed for Nebular-Chain

def extract_data_from_log(fileNames: List[str], attrName: str,
                          cut: slice = None):
    ay_list = []
    max_len = 0
    for filename in fileNames:
        with open(filename, 'r') as f:
            data = f.readlines()
            data = [eval(row.replace('\n', '')) for row in data]
        # {"INSTRUCTION":"INSTRUCTION","cumulative_remat_counts":"1","name":"aten::add.Tensor","rid":"-4661336787295776820"}
        # {"INSTRUCTION":"CONSTANT","NAME":"dependecy counts","VALUE":0}
        ay = []
        for row in data:
            ay.append(int(row[attrName]))
        ay_list.append(ay)
        print(np.mean(ay_list[-1]), np.min(ay_list[-1]), np.max(ay_list[-1]), np.median(ay_list[-1]))
        max_len = max(max_len, len(ay_list[-1]))
    
    for l in ay_list:
        l += [0]*(max_len - len(l))

    if cut is not None:
        ay_list = [l[cut] for l in ay_list]

    return [np.array(l) for l in ay_list]


def config_backup(*style_configs):
    def fn_wrapper(func: Callable):
        def para_wrapper(*args, **kwargs):
            backups = []
            for conf in style_configs:
                backups.append(eval(f"{conf}.copy()"))
            
            func(*args, **kwargs)

            for i, conf_name in enumerate(style_configs):
                eval(f"{conf_name}.clear()")
                eval(f"{conf_name}.update(backups[i])")
        return para_wrapper
    return fn_wrapper


@config_backup('bar_style')
def plot_MO1a(filepath_tmpl: str, data_tags: str, label_tmpl: str,
              title: str = '', figsize: List[float] = [5,5]):
    bar_style['edgecolor'] = None
    bar_style['alpha'] = 0.8

    fns = [ filepath_tmpl.format(i) for i in data_tags]
    labels = [ label_tmpl.format(i) for i in data_tags]
    attr_name = 'cumulative_remat_counts' # 'cumulative_remat_counts'

    dats = extract_data_from_log(fns, attr_name)
    dats = [d[:600] for d in dats]

    fig = plt.figure(figsize=figsize) 
    ax = fig.subplots(1, 1)
    bhs = []
    for i,d in enumerate(dats):
        bhs += plot_one(ax, [d[:600]], [COLOR[i]], bar_config=(1, 1))
    
    xtks = np.arange(0, dats[0].size+1, 100)
    ytks = np.arange(0, 3001, 500)
    tick_fn = lambda x: f"{int(x)}"
    axestyle(ax, xtks, ytks, xlim=(0, 1), ylim=(0, 1), xlabel='Event index', ylabel='Recursion depth', 
             xtick_label=tick_fn, ytick_label=tick_fn, legends=labels, handles=bhs)
    figstyle(fig, None, [ax], title, title_style)
    fig.subplots_adjust(.19, .21, .95, .97)
    plt.show()
    fig.savefig(SAVE_PREFIX+'mo-1a.pdf')


@config_backup('xtick_style')
def plot_MO1b(title: str = '', figsize: List[float] = [5,5]):
    xtick_style['fontweight'] = 'regular'
    xtick_style['size'] = 9
    # legend_style['ncols'] = 2
    # annot_style['fontsize'] = 9
    raw_dat = [
        [4.27, 3.92, 2.53],
        # [10.19, 9.99, 4.99]
    ]
    dats = [ np.array(d) for d in raw_dat ]

    fig = plt.figure(figsize=figsize) 
    ax = fig.subplots(1, 1)

    hbs = plot_one(ax, dats, [COLOR[4]], plot_value=True, bar_config=(.8,.8))

    xlab = ['None', 'Random', 'Expert']
    lgd = ['Llama2-Lora', 'AlphaFold']
    axestyle(ax, np.arange(3), ax.get_yticks(), 
             ylabel='Training time (ms)',
             xtick_label=xlab,
             ytick_label=lambda x: f'{x:.3}',
            #  legends=lgd, handles=hbs,
             style=0)
    figstyle(fig, None, [ax], title, title_style)
    fig.subplots_adjust(.3, .08, .99, .95)
    plt.show()
    fig.savefig(SAVE_PREFIX+'mo-1b.pdf')


@config_backup('annot_style')
def plot_MO1d(title: str = '', figsize: List[float] = [5,5]):
    # xtick_style['fontweight'] = 'regular'
    # legend_style['ncols'] = 2
    annot_style['va'] = 'bottom'
    annot_style['color'] = 'k'
    raw_dat = [
        [2.02, 3.36, 4.27, 5.13],
        # [10.19, 9.99, 4.99]
    ]
    dats = [ np.array(d) for d in raw_dat ]

    fig = plt.figure(figsize=figsize) 
    ax = fig.subplots(1, 1)

    hbs = plot_one(ax, dats, [COLOR[3]], 'plot', plot_value=True, bar_config=(.8,.8))

    xlab = ['100%', '50%', '40%', '30%']
    axestyle(ax, np.arange(len(dats[0])), np.arange(7), 
             xlabel='Memory budgets',
             ylabel='Training time (sec)',
             xtick_label=xlab,
             ytick_label=lambda x: f'{int(x)}',
             xlim=(-.1,1.1),
             style=0)
    figstyle(fig, None, [ax], title, title_style)
    fig.subplots_adjust(.16, .22, .99, .95)
    plt.show()
    fig.savefig(SAVE_PREFIX+'mo-1d.pdf')


@DeprecationWarning
# @config_backup('annot_style')
def plot_MO2a(title: str = '',
              figsize=[5,3]):
    # annot_style['fontsize'] = 12

    raw_dat = [
        [48.293, 77.176],
        [29.958, 29.676],
    ]
    dats = [np.array(d) for d in raw_dat]
    lgd = ['Backward', 'Forward']
    xlab = ['Full Caching', 'Recomputation']

    fig = plt.figure(figsize=figsize, layout='tight') 
    ax = fig.subplots(1, 1)
    bhs = []

    for i,d in enumerate(dats):
        bhs += plot_one(ax, [d], [COLOR[1-i]], bar_config=(0.4, 0.4), plot_value=True)

    axestyle(ax, np.arange(2), ax.get_yticks()[:-1],
             ylabel='Execution time (ms)',
             xtick_label=xlab,
             ytick_label=lambda x: f'{int(x)}',
             legends=lgd, handles=bhs,
             style=0
             )
    
    # figstyle(fig, None, [ax], title, title_style)
    plt.show()
    fig.savefig(SAVE_PREFIX+'mo-2a.pdf')


@config_backup('xtick_style', 'ytick_style', 'title_style', 'ylabel_style')
def plot_MO2(title: str = '',
              figsize=[5,3]):
    # xtick_style['size']=10
    # ytick_style['size']=10
    # title_style['size']=10
    # ylabel_style['size']=10

    fig, axes = plt.subplots(1, 2, figsize=figsize, layout='constrained') 

    raw_dat = [
        [38.64, 34.27, 31.34],
        [39.72, 31.14, 20.01],
    ]
    dats = [np.array(d) for d in raw_dat]
    xlab = [
        [f'PP={x}' for x in [1,2,4]],
        [f'TP={x}' for x in [1,2,4]]
    ]
    titles = ['TP=2', 'PP=4']

    for i, d in enumerate(dats):
        ax = axes[i]
        plot_one(ax, [d], [COLOR[1-i]], bar_config=(0.9, 0.9), plot_value=True)

        axestyle(ax, np.arange(len(dats[0])), ax.get_yticks()[:-1],
                ylabel='Throughput (samples/s)' if i==0 else '',
                xtick_label=xlab[i],
                ytick_label=lambda x: f'{int(x)}' if i==0 else None,
                style=0
                )
    
    figstyle(fig, None, axes, titles, title_style)
    plt.show()
    fig.savefig(SAVE_PREFIX+'tppp-var.pdf')

    # raw_dat = [
    #     [6.71, 326.50, 406.50]
    #     [3.27, 84.13, 196.64],
    # ]
    # dats = [np.array(d) for d in raw_dat]
    # xlab = ['GEMM', 'NCCL1', 'NCCL2']


@config_backup('xtick_style')
def plot_overall(title='', figsize=[5,3]):
    xtick_style['size'] = 9
    xtick_style['fontweight'] = 'regular'
    df = open_excel_with_pandas(DATA_PATH, 'overall-e2e')
    df = df.iloc[:-1, :]
    xlab, lgds, dats = extract_label_and_array(df)
    s1 = np.array([0,1,2,4,5])
    s2 = np.array([3,6])

    fig, axes = plt.subplots(1, 2, figsize=figsize, layout='tight',
                           gridspec_kw={'width_ratios': [5,2]})
    
    bhs = plot_one(axes[0], dats, COLOR, bar_config=(.8, .8), select=s1)
    plot_one(axes[1], dats, COLOR, bar_config=(.8, .8), select=s2)

    axestyle(axes[0], np.arange(len(dats[0])-2), np.arange(0, 201, 40),
             ylabel='Throughput (samples/s)',
             xtick_label=xlab[s1],
             ytick_label=lambda x: f'{int(x)}',
             legends=lgds, handles=bhs
             )

    axestyle(axes[1], np.arange(2), axes[1].get_yticks(),
            xtick_label=xlab[s2],
            ytick_label=lambda x: f'{int(x)}',
            # legends=lgds, handles=bhs
            )
    # ax.set_yscale('log')
    figstyle(fig, 'x', axes, 'Models', xlabel_style, True)
    fig.subplots_adjust(left=-0.1, right=1.1, top=1.1, bottom=-0.1)
    plt.show()
    fig.savefig(SAVE_PREFIX+'overall-e2e.pdf', bbox_inches='tight')


@config_backup('xtick_style')
def plot_pt_pp_var(title = '', figsize=[5,3]):
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout='constrained', 
                             gridspec_kw={'width_ratios':[1.3, 1, 1.1]})
    bhs = []
    
    xtitle = ['', 'Micro batch size', 'Global batch size']
    for idx, sht in enumerate(['tp-pp-var', 'mbatch-var', 'gbatch-var']):
        df = open_excel_with_pandas(DATA_PATH, sht, None)
        df = df.iloc[:-2, :]
        if sht == 'tp-pp-var':
            # xtick_style['rotation'] = 20
            xlab = [ f"TP={int(df.loc[r, 'TP size'])}\nPP={int(df.loc[r, 'PP size'])}" for r in df.index]
        elif sht == 'mbatch-var':
            # xtick_style['rotation'] = 0
            xlab = [ f"{int(df.loc[r, 'microbatch size'])}" for r in df.index]
        elif sht == 'gbatch-var':
            # xtick_style['rotation'] = 0
            xlab = [ f"{int(df.loc[r, 'global_batch'])}" for r in df.index]
        df = df.iloc[:, 6:]
        _, lgds, dats = extract_label_and_array(df)

        bhs += plot_one(axes[idx], dats, [COLOR[-1], COLOR[-2]], bar_config=(.8, .8))
        if idx == 0:
            axestyle(axes[idx], np.arange(len(xlab)), axes[idx].get_yticks(),
                    xlabel=xtitle[idx],
                    ylabel='Throughput (samples/s)',
                    xtick_label=xlab,
                    ytick_label=lambda x: f'{int(x)}',
                    # ylim=(0, 1.25),
                    legends=lgds, handles=bhs)
        else:
            axestyle(axes[idx], np.arange(len(xlab)), axes[idx].get_yticks()[:-1],
                    xlabel=xtitle[idx],
                    xtick_label=xlab,
                    ytick_label=lambda x: f'{int(x)}')

    plt.show()
    fig.savefig(SAVE_PREFIX+'var.pdf')


@config_backup('ylabel_style', 'legend_style')
def plot_scalability(figsize=[5,3]):
    legend_style['ncols'] = 5
    legend_style['prop']['size'] = 10
    ylabel_style['size'] = 14
    fig, axes = plt.subplots(3, 2, figsize=figsize, layout='constrained')
    flat_axes = flattern(axes)
    
    title = ['GPT3-1.7B', 'Llama2-7B',
             'GPT3-7.5B', 'Llama2-13B',
             'GPT3-121B', 'Llama2-70B']
    ytksz = [150,  50, 40, 20, 2, 3, ]
    bhs = None
    for idx, key in enumerate([t.lower() for t in title]):

        df = open_excel_with_pandas(DATA_PATH, f'scale-{key}')
        df = df.T
        _, lgds, dats = extract_label_and_array(df)
        for i in range(len(lgds)):
            if lgds[i] == 'DT-Control' or lgds[i] == 'Nebula-Chain':
                lgds[i] = OURS
        # xlab = ['DZ', 'M-LM', 'M-DP', 'NC']
        xlab = [f"{pow(2, x)}" for x in range(4,9)]
        c = COLOR.copy()[:5]
        tmp = c[-1]
        c[-1] = c[-2]
        c[-2] = tmp
        bhs = plot_one(flat_axes[idx], dats, c, bar_config=(.9, .9), invalid_val=0, invalid_text='OOM')
        if idx%2 == 0:
            axestyle(flat_axes[idx], np.arange(len(dats[0])), np.arange(6)*ytksz[idx],
                    # ylabel='Throughput (samples/s)',
                    # xlabel=title[idx],
                    xtick_label=xlab,
                    ytick_label=lambda x: f'{int(x)}',
                    # ylim=(1, 1.3),
                    # legends=[f'{x} GPUs' for x in lgds], handles=bhs
                    )
            flat_axes[idx].set_title(title[idx])
        else:
            axestyle(flat_axes[idx], np.arange(len(dats[0])), np.arange(6)*ytksz[idx],
                # xlabel=title[idx],
                xtick_label=xlab,
                ytick_label=lambda x: f'{int(x)}',
                # ylim=(1, 1.3),
                ) 
            flat_axes[idx].set_title(title[idx])
            
    flat_axes[-2].set_xlabel('Number of GPUs')
    flat_axes[-1].set_xlabel('Number of GPUs')
    # grid_legend_external(flat_axes[-1], lgds, bhs)
    fig.legend(bhs, lgds, loc='outside upper center', **legend_style)
    figstyle(fig, 'y', axes, 'Throughput (samples/s)' ,ylabel_style, True)

    # fig.subplots_adjust(.08, .08, .995, .95, .1, .6)
    plt.show()
    fig.savefig(SAVE_PREFIX+'exp_scale.pdf')


@config_backup('xtick_style', 'annot_style')
def plot_remat_counts(title='', figsize=[5,3]):
    xtick_style['rotation']=30
    annot_style['ha'] = 'left'
    annot_style['va'] = 'center'
    annot_style['color'] = 'k'
    # annot_style['fontweight'] = 'regular'
    annot_style['fontsize'] = 8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, layout='constrained',
                                   gridspec_kw={'width_ratios': [4, 4]})
    df = open_excel_with_pandas(DATA_PATH, 'remat-count')
    xlab, lgds, dats = extract_label_and_array(df.T)
    for i in range(len(lgds)):
        if 'NC' in lgds[i]:
            lgds[i] = lgds[i].replace('NC', 'Ours')
    dats = [np.flip(x)[:-1] for x in dats]

    bhs = plot_one(ax1, dats, COLOR, 'barh', bar_config=(.6,.6), 
                   anno_fn=lambda x: f'{float(x):.3}')
    bhs = plot_one(ax2, dats, COLOR, 'barh', bar_config=(.6,.6), 
                   anno_fn=lambda x: f'{float(x):.3}')
    axestyle(ax1, np.arange(5)*1e5, np.arange(len(dats[0])),
            #  xlabel='Event Counts',
             ylabel='Memory Budget Ratios',
             xtick_label=lambda x: f'{int(x/1e3)}K' if x > 0 else '0',
             ytick_label=np.flip(xlab)[:-1],
            #  handles=bhs, legends=lgds,
             style=1)
    
    axestyle(ax2, np.arange(5)*4e5+6e5, np.arange(len(dats[0])),
            #  xlabel='Event Counts',
            #  ylabel='Memory Budget Ratios',
             xtick_label=lambda x: f'{round(x/1e6, 2)}M' if x > 0 else '0',
             ytick_label=['']*(len(xlab)-1),
             handles=bhs, legends=lgds,
             style=1)
    
    ax1.set_xlim(0, 4.5e5)
    ax2.set_xlim(5e5, 2.2e6)
    ax1.spines.right.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.tick_params(axis='y', size=0)

    d = 1  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=2, clip_on=False)
    ax1.plot([1, 1], [1, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 0], [1, 0], transform=ax2.transAxes, **kwargs)

    figstyle(fig, 'x', (ax1, ax2), 'Event Counts', xlabel_style, True)

    plt.show()
    fig.savefig(SAVE_PREFIX+'exp_remat-count.pdf')
    pass

@config_backup('legend_style')
def plot_tt_with_budegts(figsize=[5,3]):
    legend_style['ncols'] = 4
    legend_style['prop']['size'] = 12
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout='constrained')
    
    title = ['Llama2-7B Lora', 'AlphaFold', 'GPT3-7.5B']
    title_sup = ['8GPU (80GB nvlink)', '8GPU (80GB nvlink)', '64GPU (40GB PCIe)']
    g_lgds = np.array([])
    g_bhs = []
    for idx, key in enumerate([t.lower() for t in title]):
        df = open_excel_with_pandas(DATA_PATH, f'compress-tt-{key}')
        df = df.T
        xlab, lgds, dats = extract_label_and_array(df)
        if idx != 1:
            g_lgds = np.concatenate((g_lgds, lgds))
        ax = axes[idx]
        c = COLOR if idx < 2 else COLOR[5:] + COLOR[3:]
        bhs = plot_one(ax, dats, c, bar_config=(.9, .9), invalid_val=0., invalid_text='X')
        if idx == 0:
            g_bhs += bhs
        elif idx == 2:
            g_bhs += bhs[:-2]
        axestyle(ax, np.arange(len(dats[0])), ax.get_yticks(),
                ylabel='Training time (s/iter)' if idx == 0 else '',
                xlabel='Memory Budget Ratios',
                xtick_label=xlab,
                ytick_label=lambda x: f'{int(x)}' if x < 1000 else f'{int(x/1000)}K',
                # legends=lgds, handles=bhs
                )
        ax.set_title(title[idx] + " " + title_sup[idx])
    fig.legend(g_bhs, g_lgds, loc='outside upper center', **legend_style)
    
    plt.show()
    fig.savefig(SAVE_PREFIX+'budget-tt.pdf')


@config_backup('legend_style')
def plot_mu_frag_with_budegts_(figsize=[5,3]):
    legend_style['ncols'] = 5
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout='constrained')
    
    title = ['Llama2-7B Lora', 'AlphaFold']
    bhs = None

    def tick_filter(dats, ticks):
        ret = ticks
        for i in range(ticks.size):
            if i > 0 and max(flattern(dats)) < ticks[i-1] or ticks[i] < 0:
                ret = np.delete(ticks, i)
        return ret

    for idx, key in enumerate([t.lower() for t in title]):
        mu_df = open_excel_with_pandas(DATA_PATH, f'compress-mu-{key}').T
        frag_df = open_excel_with_pandas(DATA_PATH, f'compress-frag-{key}').T

        mu_xlab, mu_lgds, mu_dats = extract_label_and_array(mu_df)
        frag_xlab, frag_lgds, frag_dats = extract_label_and_array(frag_df)
        mu_ax = axes[idx]
        frag_ax = mu_ax.twinx()
        bhs = plot_one(mu_ax, mu_dats, COLOR, bar_config=(.8, .8), invalid_val=0., invalid_text='FAILED')
        plot_one(frag_ax, frag_dats, COLOR, 'plot', bar_config=(.9, .9), invalid_val=0.)
        
        mu_tk = tick_filter(mu_dats, mu_ax.get_yticks())
        frag_tk = tick_filter(frag_dats, frag_ax.get_yticks())
        mu_step = mu_tk[1] - mu_tk[0]
        frag_step = frag_tk[1] - frag_tk[0]
        mu_size = mu_tk.size
        frag_size = frag_tk.size
        mu_tks = np.concatenate((mu_tk, np.array([mu_tk[-1]+mu_step*i for i in range(1, frag_size)])))
        frag_tks = np.concatenate((np.array([x*frag_step for x in range(-1, -mu_size, -1)]), frag_tk))
        axestyle(mu_ax, np.arange(len(mu_dats[0])), mu_tks,
                ylabel='Peak Reserve Memory (MB)' if idx == 0 else '',
                xlabel='Memory Budget Ratios',
                ytick_label=lambda x: f'{x:.3}' if x <= mu_tk[-1] else ''
                )
        axestyle(frag_ax, np.arange(len(frag_dats[0])), frag_tks,
                ylabel='Fragment ratio' if idx == 1 else '',
                xtick_label=frag_xlab,
                ytick_label=lambda x: f'{abs(round(x, 3))}' if x >= -1e-4 else '',
                style=2
                )
        mu_ax.set_title(title[idx])
    fig.legend(bhs, mu_lgds, loc='outside upper center', **legend_style)
    
    plt.show()
    fig.savefig(SAVE_PREFIX+'budget-mu-frag.pdf')



@config_backup('legend_style')
def plot_frag_with_budegts(figsize=[5,3]):
    legend_style['ncols'] = 4
    legend_style['prop']['size'] = 12
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout='constrained')
    
    title = ['Llama2-7B Lora', 'AlphaFold', 'GPT3-7.5B']
    title_sup = ['8GPU (80GB nvlink)', '8GPU (80GB nvlink)', '64GPU (40GB PCIe)']
    g_lgds = np.array([])
    g_bhs = []

    def tick_filter(dats, ticks):
        ret = ticks
        for i in range(ticks.size):
            if i > 0 and max(flattern(dats)) < ticks[i-1] or ticks[i] < 0:
                ret = np.delete(ticks, i)
        return ret

    for idx, key in enumerate([t.lower() for t in title]):
        frag_df = open_excel_with_pandas(DATA_PATH, f'compress-frag-{key}').T

        frag_xlab, frag_lgds, frag_dats = extract_label_and_array(frag_df)
        if idx != 1:
            g_lgds = np.concatenate((g_lgds, frag_lgds))

        frag_ax = axes[idx]
        c = COLOR if idx < 2 else COLOR[5:] + COLOR[3:]
        bhs = plot_one(frag_ax, frag_dats, c, 'plot', bar_config=(.9, .9), invalid_val=0.)
        if idx == 0:
            g_bhs += bhs
        elif idx == 2:
            g_bhs += bhs[:-2]
        
        # ylabel_style['size'] -=1
        # axestyle(mu_ax, np.arange(len(mu_dats[0])), tick_filter(mu_dats, mu_ax.get_yticks()),
        #         ylabel='Peak Reserve Memory (MB)' if idx == 0 else '',
        #         xlabel='Memory Budget Ratios',
        #         xtick_label=mu_xlab,
        #         ytick_label=lambda x: '0' if x==0 else f'{int(x/1e3)}K',
        #         # legends=mu_lgds, handles=bhs,
        #         )
        # ylabel_style['size'] +=1
        axestyle(frag_ax, np.arange(len(frag_dats[0])), tick_filter(frag_dats, frag_ax.get_yticks()),
                ylabel='Fragment Ratio' if idx == 0 else '',
                xtick_label=frag_xlab,
                ytick_label=lambda x: f'{abs(round(x, 3))}',
                # legends=frag_lgds, handles=bhs,
                )
        frag_ax.set_title(title[idx] + ' ' + title_sup[idx])
    fig.legend([x[0] for x in g_bhs], g_lgds, loc='outside upper center', **legend_style)
    
    plt.show()
    fig.savefig(SAVE_PREFIX+'budget-frag.pdf')



@config_backup('legend_style')
def plot_mu_with_budegts(figsize=[5,3]):
    legend_style['ncols'] = 4
    legend_style['prop']['size'] = 12
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout='constrained')
    
    title = ['Llama2-7B Lora', 'AlphaFold', 'GPT3-7.5B']
    title_sup = ['8GPU (80GB nvlink)', '8GPU (80GB nvlink)', '64GPU (40GB PCIe)']
    g_lgds = np.array([])
    g_bhs = []

    def tick_filter(dats, ticks):
        ret = ticks
        for i in range(ticks.size):
            if i > 0 and max(flattern(dats)) < ticks[i-1] or ticks[i] < 0:
                ret = np.delete(ticks, i)
        return ret

    for idx, key in enumerate([t.lower() for t in title]):
        mu_df = open_excel_with_pandas(DATA_PATH, f'compress-mu-{key}').T

        mu_xlab, mu_lgds, mu_dats = extract_label_and_array(mu_df)
        if idx != 1:
            g_lgds = np.concatenate((g_lgds, mu_lgds))

        mu_ax = axes[idx]
        c = COLOR if idx < 2 else COLOR[5:] + COLOR[3:]
        bhs = plot_one(mu_ax, mu_dats, c, bar_config=(.8, .8), invalid_val=0., invalid_text='X')
        if idx == 0:
            g_bhs += bhs
        elif idx == 2:
            g_bhs += bhs[:-2]
        
        ylabel_style['size'] -=1
        axestyle(mu_ax, np.arange(len(mu_dats[0])), tick_filter(mu_dats, mu_ax.get_yticks()),
                ylabel='Peak Reserve Memory (MB)' if idx == 0 else '',
                xlabel='Memory Budget Ratios',
                xtick_label=mu_xlab,
                ytick_label=lambda x: '0' if x==0 else f'{int(x/1e3)}K',
                # legends=mu_lgds, handles=bhs,
                )
        ylabel_style['size'] +=1

        mu_ax.set_title(title[idx] + ' ' + title_sup[idx])
    fig.legend(g_bhs, g_lgds, loc='outside upper center', **legend_style)
    
    plt.show()
    fig.savefig(SAVE_PREFIX+'budget-mu.pdf')


@config_backup('bar_style')
def plot_cumulative_remat_counts(filepath_tmpl: str, data_tags: str,
              title: str = '', figsize: List[float] = [5,5]):
    bar_style['edgecolor'] = None
    bar_style['alpha'] = 0.8

    fns = [ filepath_tmpl.format(i) for i in data_tags ]
    labels = [ f'{i} 30% budgets' for i in ['DTR', 'Ours']]
    attr_name = 'cumulative_remat_counts' # 'cumulative_remat_counts'

    dats = extract_data_from_log(fns, attr_name)
    dats = [d[:600] for d in dats]

    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=figsize, layout='constrained',
                                   gridspec_kw={'height_ratios': [2, 3]}) 
    bhs = []
    cs = [COLOR[4], COLOR[3]]
    for i,d in enumerate(dats):
        bhs += plot_one(ax1, [d[:600]], [cs[i]], bar_config=(1, 1))
        plot_one(ax2, [d[:600]], [cs[i]], bar_config=(1, 1))
    
    xtks = np.arange(0, dats[0].size+1, 100)
    ytks1 = np.arange(0, 201, 50)
    ytks2 = np.arange(1000, 3000+1, 1000)
    tick_fn = lambda x: f"{int(x)}"
    axestyle(ax1, xtks, ytks1, xlim=(0, 1), xlabel='Event index',
             xtick_label=tick_fn, ytick_label=tick_fn)
    axestyle(ax2, xtks, ytks2, xlim=(0, 1),  
            xtick_label=lambda x:'', ytick_label=tick_fn, legends=labels, handles=bhs)
    
    figstyle(fig, None, [ax1, ax2], title, title_style)

    ax1.set_ylim(0, 220)
    ax2.set_ylim(220, 3000)
    ax1.spines.top.set_visible(False)
    ax2.spines.bottom.set_visible(False)
    ax2.tick_params(axis='x', size=0)

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=2, clip_on=False)
    ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
    figstyle(fig, 'y', [ax1, ax2], 'Recursion depth', ylabel_style, True)

    # fig.subplots_adjust(.19, .21, .95, .97)
    plt.show()
    fig.savefig(SAVE_PREFIX+'exp_cumulative-remat-count.pdf')


def plot_budgt_overhead(figsize=[5,3]):
    fig, ax = plt.subplots(1,1, figsize=figsize, layout='constrained')
    
    df = open_excel_with_pandas(DATA_PATH, 'budget-overhead').T
    xlab, lgds, dats = extract_label_and_array(df)

    bhs = plot_one(ax, dats, COLOR[-3:], 'plot')
    axestyle(ax, np.arange(len(xlab)), ax.get_yticks(),
             xlabel='Memory Budget Ratios',
             ylabel='Computation Overhead',
             xtick_label=xlab,
             ytick_label=lambda y: f'{round(y, 3):.2f}',
             legends=lgds
             )
    # ax.legend(lgds, **legend_style)
    plt.show()
    fig.savefig(SAVE_PREFIX+'memory-budget-overhead.pdf')



@config_backup('bar_style', 'legend_style')
def plot_bw_comm_var(figsize=[5,5]):
    bar_style['edgecolor'] = None
    # legend_style['ncols'] = 2
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout='constrained')

    tits = ['Backward Time', 'Communication Time']
    gxlab = ['Micro-batch Index', 'Communication Index']
    ymin = [400, 0]
    for idx, sht in enumerate(['backward-var', 'comm-var']):
        df = open_excel_with_pandas(DATA_PATH, sht, index_col=None)
        
        _, lgds, dats = extract_label_and_array(df)
        ax = axes[idx]
        hdl=plot_one(ax, [dats[0]], [COLOR[0]], bar_config=(.6,.6))
        # plot_one(ax, [dats[1]], [COLOR[3]], 'plot', markers=[','])
        axestyle(ax, np.arange(0, dats[0].size, 25), ax.get_yticks(),
                xlabel=gxlab[idx],
                xlim=(0,1),
                xtick_label=lambda x: f'{int(x)}',
                ytick_label=lambda x: f'{int(x)}',
                legends=lgds[0:1] if idx == 1 else None,
                handles=hdl,
                )
        ax.set_title(tits[idx])
        ax.set_ylim(ymin[idx], ax.get_yticks()[-1])
        ax.tick_params(length=0)

    figstyle(fig, 'y', axes, 'Time (ms)', ylabel_style)
    # fig.legend(np.flip(lgds), loc='outside upper center', **legend_style)
    plt.show()
    fig.savefig(SAVE_PREFIX+'bw-comm-var.pdf')
        

if __name__ == "__main__":
    # @Note 重物化过深问题
    # plot_MO1a('./logs/remat/remat_counts_{}0%.log', [3,4,5], '{}0% budget', 
    #         #   'Cumulatvie Remat Counts', 
    #           figsize=[3,2])
    # plot_MO1b(
    #     # 'Training Overhead Under 30% Memory Budget', 
    #     figsize=[.5, 2])
    # plot_MO1d(
    #     # 'Training Overhead Under 30% Memory Budget', 
    #     figsize=[.5, 2])

    # plot_MO2()

    # @Note 实验部分
    # plot_overall(figsize=[4.2,2.5])
    # plot_pt_pp_var(figsize=[7.5, 2.5])
    plot_scalability(figsize=[9,6])
    plot_remat_counts('Eviction and rematerialization counts', figsize=[5,3.5])
    plot_cumulative_remat_counts('./logs/remat/remat_{}counts_30%.log', ['', 'nc_'], figsize=[5,3])
    # plot_tt_with_budegts()
    # plot_mu_frag_with_budegts(figsize=[6,6])
    # plot_frag_with_budegts(figsize=[6,6])
    # plot_mu_with_budegts(figsize=[6,6])
    # plot_budgt_overhead()
    # plot_bw_comm_var()