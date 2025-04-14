import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
# from tools import csv2numpy, find_all_files, group_files
import argparse

# Define parser
parser = argparse.ArgumentParser()
# Add Directory Name
parser.add_argument(
    '--directory_name', type=str, default=1, help='directory name (default: 0)')
# Add Cols Name
parser.add_argument(
    '--cols_name', type=str, default=1, help='choose cols to draw (default: 0)')
# Add ylabel Name
parser.add_argument(
    '--ylabel_name', type=str, default=1, help='ylabel name (default: 0)')


def get_filename(file_name, algorithm):
    path = []
    for p in range(5):
        path.append(os.path.join(file_name, algorithm, '{}'.format(p), 'progress.csv'))

    return path


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''Smooth signal y, where radius is determines the size of the window.

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-0)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / \
            np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / \
            np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


COLORS = (
    [
        # deepmind style
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        # '#F0E442',
        '#d73027',  # RED
        # built-in color
        'blue',
        'red',
        'pink',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'purple',
        'brown',
        'orange',
        'teal',
        'lightblue',
        'lime',
        'lavender',
        'turquoise',
        'darkgreen',
        'tan',
        'salmon',
        'gold',
        'darkred',
        'darkblue',
        'green',
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ]
)

COLORS=(
    [
        'blue',
        '#ffc832',  # YELLOW
        '#f781bf',  # PINK
        '#984ea3',  # PURPLE
        '#4daf4a',  # GREEN
        '#f46d43',  # ORANGE
        'brown',
        'turquoise',
    ]
)

def plot_ax(
    ax,
    file_lists,
    legend_pattern=".*",
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    xkey='env_step',
    ykey='rew',
    smooth_radius=0,
    shaded_std=True,
    legend_outside=False,
):

    def legend_fn(x):
        # return os.path.split(os.path.join(
        #     args.root_dir, x))[0].replace('/', '_') + " (10)"
        return re.search(legend_pattern, x).group(0)

    legneds = map(legend_fn, file_lists)
    # sort filelist according to legends
    file_lists = [f for _, f in sorted(zip(legneds, file_lists))]
    legneds = list(map(legend_fn, file_lists))

    # upper letters
    legneds = [legneds[i].upper() for i in range(len(legneds))]

    for index, csv_file in enumerate(file_lists):
        csv_dict = csv2numpy(csv_file)
        x, y = csv_dict[xkey], csv_dict[ykey]
        # x = x[:6000]
        # y = y[:6000]
        y = smooth(y, radius=smooth_radius)
        color = COLORS[index % len(COLORS)]
        ax.plot(x, y, color=color, linewidth=1)

    ax.legend(
        legneds,
        loc=2 if legend_outside else "upper left",
        bbox_to_anchor=(1, 1) if legend_outside else None,
        fontsize = 12,
    )

    for index, csv_file in enumerate(file_lists):
        csv_dict = csv2numpy(csv_file)
        x, y = csv_dict[xkey], csv_dict[ykey]
        # x = x[:6000]
        # y = y[:6000]
        y = smooth(y, radius=smooth_radius)
        color = COLORS[index % len(COLORS)]
        if shaded_std and ykey + ':shaded' in csv_dict:
            y_shaded = smooth(csv_dict[ykey + ':shaded'], radius=smooth_radius)
            # y_shaded = y_shaded[:5000]
            ax.fill_between(x, y - y_shaded, y + y_shaded, color=color, alpha=.2)

    ax.xaxis.set_major_formatter(mticker.EngFormatter())
    if xlim is not None:
        ax.set_xlim(xmin=0, xmax=xlim)
    # add title
    ax.set_title(title)
    ax.grid(True)
    # add labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def plot_figure(
    file_lists,
    group_pattern=None,
    fig_length=6,
    fig_width=6,
    sharex=False,
    sharey=False,
    title=None,
    **kwargs,
):
    if not group_pattern:
        fig, ax = plt.subplots(figsize=(fig_length, fig_width))
        plot_ax(ax, file_lists, title=title, **kwargs)
    else:
        res = group_files(file_lists, group_pattern)
        row_n = int(np.ceil(len(res) / 3))
        col_n = min(len(res), 3)
        fig, axes = plt.subplots(
            row_n,
            col_n,
            sharex=sharex,
            sharey=sharey,
            figsize=(fig_length * col_n, fig_width * row_n),
            squeeze=False
        )
        axes = axes.flatten()
        for i, (k, v) in enumerate(res.items()):
            plot_ax(axes[i], v, title=k, **kwargs)
    if title:  # add title
        fig.suptitle(title, y=0.93, fontsize=13)


if __name__ == "__main__":
    args = parser.parse_args()

    env_list = ['Antcost-v0','Cartpolecost-v0','HalfCheetahcost-v0','Humanoidcost-v0',
                'Minitaurcost-v0','Pointcirclecost-v0','Spacereachcost-v0','Spacerandomcost-v0','Swimmercost-v0','Spacedualarmcost-v0']

    step_dic = {
        'Antcost-v0': 1.e6,
        'Cartpolecost-v0': 3.e5,
        'HalfCheetahcost-v0': 5.e5,
        'Humanoidcost-v0': 1.e6,
        'Minitaurcost-v0': 1.e6,
        'Pointcirclecost-v0': 3.e5,
        'Spacereachcost-v0': 3.e5,
        'Spacerandomcost-v0': 5.e5,
        'Swimmercost-v0': 3.e5,
        'Spacedualarmcost-v0': 5.e5,

    }

    for env in env_list:
        # Set random seed
        args.directory_name = './log_plot/{}'.format(env)
        args.cols_name = 'return'
        args.ylabel_name = 'Cost Return'

        args.maxsteps = step_dic[env]
        # alg = 'HalfCheetahcost-v0'
        # alg = 'Minitaurcost'
        # alg = 'swimmer'
        algorithm_list = ['ALAC', 'LAC', 'LAC*', 'LBPO', 'POLYC', 'SAC_cost', 'SPPO', 'TNLF']

        path_all = []
        for num in range(len(algorithm_list)):
            path = get_filename(args.directory_name, algorithm_list[num]).copy()
            path_all.append(path)

        mean_all = []  # 不同随机种子下的均值的均值
        std_all = []
        steps_all = []
        smooth_radius = 10
        order = 0


        for pt in path_all:
            record = []  # 每个算法用空record
            col_values = []
            steps = []

            for rnd in range(5):
                record.append(pd.read_csv(pt[rnd]))
                col_values.append(record[rnd][args.cols_name])
                col_values[rnd] = smooth(col_values[rnd], radius=smooth_radius, mode='two_sided')

            # verify the timesteps
            if 'LBPO' in pt[0]:
                steps.append(record[0]['timesteps'].to_list())
            else:
                steps.append(record[0]['total_timesteps'].to_list())

            for idx,t in enumerate(steps[-1]):
                if t > args.maxsteps:
                    break
            max_t = idx

            record_mean = []
            record_std = []
            t = col_values[0].shape[0]
            for j in range(t):  # 遍历timestep求均值和方差
                temp_record = np.zeros(5)
                for rnd in range(5):
                    temp_record[rnd] = col_values[rnd][j]

                record_mean.append(temp_record.mean())
                record_std.append(temp_record.std())

            mean_all.append(record_mean[:max_t])
            std_all.append(record_std[:max_t])
            steps_all.append(steps[0][:max_t])

            order += 1


        plt.figure(figsize=(6, 4), dpi=300)

        labels = ['ALAC(ours)', 'LAC', 'LAC*', 'LBPO', 'POLYC', 'SAC-cost', 'SPPO', 'TNLF']
        for i in range(len(algorithm_list)):
            color = COLORS[i % len(COLORS)]
            plt.plot(steps_all[i], mean_all[i], label=labels[i], color=color, linewidth=1.5)
            mean_sub = np. array(mean_all[i])
            std_sub = np.array(std_all[i])
            plt.fill_between(steps_all[i],
                             mean_sub - std_sub,
                             mean_sub + std_sub,
                             alpha=0.2, color=color)  # alpha=0.2是阴影区的透明度

        # plt.legend(loc="upper right", prop={'size': 6})
        legend_outside = False
        plt.legend(
            loc=2 if legend_outside else "upper right",
            bbox_to_anchor=(1, 1) if legend_outside else None,
            fontsize=12, prop={'size': 13}
        )
        title = args.directory_name.replace('-v0','')
        title = title.replace('./log_plot/', '')
        title = title.replace('cost', '-cost')
        plt.title(title,fontsize=16)
        plt.grid(True)
        plt.ylabel(args.ylabel_name,fontsize=16)
        plt.xlabel('Timestep',fontsize=16)
        plt.tight_layout()
        plt.ticklabel_format(style='sci', scilimits=(0, 0),axis='x')
        savepath = args.directory_name + args.cols_name + '_compare{}'.format(1)+'.pdf'
        plt.savefig(savepath, format='pdf', dpi=300)
    # plt.show()
