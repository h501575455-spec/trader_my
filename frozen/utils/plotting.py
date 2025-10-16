import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import calplot
from joypy import joyplot

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..basis import FrozenConfig
    from ..factor import Factor

from .helper import _lazy_import_factor

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Union, Dict, List

def load_style(prefer="light", module="backtest"):
    """Set plot style for figure."""
    plt.rcdefaults()
    if prefer == "light":
        light_params = {"axes.labelsize": 13, "axes.titlesize": 16}
        if module == "backtest":
            pass
        elif module == "factor":
            plt.rcParams.update(light_params)
    elif prefer == "dark":
        plt.style.use("seaborn-v0_8")
        dark_params = {
            "figure.facecolor": "#252526",
            "axes.facecolor": "#252526",
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        }
        plt.rcParams.update(dark_params)
        if module == "backtest":
            plt.rcParams.update({"axes.labelsize": 12, "axes.titlesize": 15})
        elif module == "factor":
            plt.rcParams.update({"axes.labelsize": 13, "axes.titlesize": 16})
    else:
        raise ValueError(f"The specified `{prefer}` plot style not found.")


def customize_colormap(colors: list, cmap_name: str, n_bins: int = 256):
    """
    Create a customized colormap that can adapt to any matplotlib
    functions that take `camp` as an input.

    Parameters:
    -----------
    colors: list
        User defined base color points to create the colormap.
        e.g. "crest": ["#1C3F5F", "#1D5F7C", "#2C7C95", "#4E9BAA", "#7FBCBB", "#B2DBCD"]
    cmap_name: str
        The name of customized colormap.
    n_bins: int
        The number of color levels in total.
        If large (default), create continuous color mappings.
        if small (len(colors)), create discrete color mappings.
    """
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cmap


class STGPlot:
    """The backtest visualization module."""

    def __init__(self):
        plt.rcdefaults()

    def _percentage_formatter(self, x, pos):
        return f"{x*100:.0f}%"

    def create_pnl_plot(
            self,
            port_ret: pd.DataFrame,
            date_rule: str,
            plot_type: str = None,
            dynamic: bool = False,
            mode: str = "dark"
        ):
        """
        Generate strategy performance visualization plots for a portfolio.
        It can create two types of plots:

        1. Scatter Plot:
            Displays the periodic PnL (Profit and Loss) of the account. 
            Each point represents a period's PnL, color-coded based on 
            whether it's positive (green) or negative (red).

        2. Line Plot:
            Shows the net value curves for both the account and a benchmark 
            over time. The area between the benchmark line and the baseline 
            is filled with color to highlight over/underperformance.
        
        Parameters:
        -----------
        port_ret: pd.DataFrame
            The portfolio returns of the account.
        
        date_rule: str
            The frequency of data resampling, consistent with config.

        plot_type: str
            This method allows for 3 types of plotting options.

            - "scatter": Only the PnL scatter plot.
            
            - "line": Only the net value line plot.
            
            - "all": Both plots combined in a single figure.
        
        mode: str
            The figure background mode.

            - "light": light background
            
            - "dark": dark background
        """

        net_value = (1 + port_ret).cumprod()

        load_style(mode)

        if plot_type != "all":
            fig = plt.figure(figsize=(15, 6))
            if plot_type == "scatter":
                ax1 = fig.add_subplot(1, 1, 1)
            if plot_type == "line":
                ax2 = fig.add_subplot(1, 1, 1)
        else:
            fig = plt.figure(figsize=(15, 12))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
        
        if plot_type == "all" and dynamic:

            y = net_value["Benchmark"]
            fill_region_up = np.where(y >= 1, y, 1)
            fill_region_down = np.where(y < 1, y, 1)

            period_pnl = net_value["Account"].resample(date_rule).last().pct_change().fillna(net_value["Account"]-1)
            period_pnl_full = period_pnl.reindex(net_value.index)
            
            ax1.grid(True, linestyle="-", alpha=0.1)
            ax2.grid(True, linestyle="-", alpha=0.1)
            ax1.hlines(y=0, xmin=period_pnl.index.min(), xmax=period_pnl.index.max(), linestyle="--", color="grey")
            ax2.hlines(y=1, xmin=net_value.index.min(), xmax=net_value.index.max(), linestyle="--", color="grey")

            ax2.fill_between(x=net_value.index, y1=1, y2=fill_region_up, color="lightcoral", alpha=0.4)
            ax2.fill_between(x=net_value.index, y1=1, y2=fill_region_down, color="lightblue", alpha=0.4)

            scatter = ax1.scatter([], [], s=50, c=[], cmap="RdYlGn_r", marker="o", edgecolors="none", alpha=0.7, vmin=0, vmax=1)
            line1, = ax2.plot([], [], linewidth=1.5, color="#FEE0D2", label="Account")
            line2, = ax2.plot([], [], linewidth=1.5, color="grey", label="Benchmark")
            frame_text = ax1.text(0.02, 0.9, "", transform=ax1.transAxes, fontsize=10, ha="left", va="center", bbox=dict(boxstyle="round", facecolor="#d1c6bf", pad=1.0, alpha=0.5))

            # ax1.set_xlim(net_value.index[0], net_value.index[-1])
            # ax2.set_xlim(net_value.index[0], net_value.index[-1])
            ax1.set_ylim(period_pnl.min() - 0.01, period_pnl.max() + 0.01)
            ax2.set_ylim(min(net_value.min()) * 0.95, max(net_value.max()) * 1.05)

            ax1.set_title("Trade PnL")
            ax2.set_title("Portfolio Performance")
            ax1.set_ylabel("PnL")
            ax2.set_ylabel("Net Value")
            ax1.set_xticks([])
            ax2.set_xlabel("Trade Date")
            ax2.legend(loc="upper left", fontsize=12)

            # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            # ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            # plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            ax1.yaxis.set_major_formatter(FuncFormatter(self._percentage_formatter))

            scatter_data = {
                "dates": [],
                "values": [],
                "colors": []
            }
            line_data = {
                "dates": [],
                "account": [],
                "benchmark": []
            }

            def update(frame):
                # update text message
                info_text = f'Date: {net_value.index[frame].strftime("%Y-%m-%d")}\n'
                info_text += f'Daily PnL: {port_ret["Account"].iloc[frame]*100:.2f}%\n'
                info_text += f'Account Value: {net_value["Account"].iloc[frame]:.2f}'
                frame_text.set_text(info_text)

                # update scatter point
                scatter_data["dates"].append(period_pnl_full.index[frame])
                current_pnl = period_pnl_full.iloc[frame]
                scatter_data["values"].append(current_pnl)
                scatter_data["colors"].append(1 if current_pnl > 0 else 0)
                
                # update line
                line_data["dates"].append(net_value.index[frame])
                line_data["account"].append(net_value["Account"].iloc[frame])
                line_data["benchmark"].append(net_value["Benchmark"].iloc[frame])
                
                scatter.set_offsets(np.c_[mdates.date2num(scatter_data["dates"]), scatter_data["values"]])
                scatter.set_array(np.array(scatter_data["colors"]))

                line1.set_data(line_data["dates"], line_data["account"])
                line2.set_data(line_data["dates"], line_data["benchmark"])
                
                return scatter, line1, line2, frame_text
            
            ani = FuncAnimation(fig, update, frames=len(net_value), interval=10, blit=True, repeat=False)
    
            # plt.tight_layout()
            plt.show()

        # PnL scatter plot
        if plot_type in ["scatter", "all"]:

            period_pnl = net_value["Account"].resample(date_rule).last().pct_change().fillna(net_value["Account"]-1)
            category = np.where(period_pnl > 0, 1, 0)
            ax1.grid(True, linestyle="-", alpha=0.1)
            ax1.hlines(y=0, xmin=period_pnl.index.min(), xmax=period_pnl.index.max(), linestyle="--", color="grey")

            if dynamic:

                scatter = ax1.scatter([], [], s=50, c=[], cmap="RdYlGn_r", marker="o", edgecolors="none", alpha=0.7, vmin=0, vmax=1)
                dates_list = []
                pnl_list = []
                category_data = []

                def init():
                    ymin = period_pnl.min() - 0.01
                    ymax = period_pnl.max() + 0.01
                    ax1.set_ylim(ymin, ymax)
                    return scatter,

                def update(frame):
                    date, pnl = frame
                    dates_list.append(date)
                    pnl_list.append(pnl)
                    category_data.append(1 if pnl > 0 else 0)
                    # Update scatters
                    scatter.set_offsets(np.c_[mdates.date2num(dates_list), pnl_list])
                    scatter.set_array(np.array(category_data))
                    return scatter,
                
                frames = list(zip(period_pnl.index, period_pnl.values))
                ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=10, repeat=False)
            else:
                scatter = ax1.scatter(period_pnl.index, period_pnl, s=50, c=category, cmap="RdYlGn_r", marker="o", edgecolors="none", alpha=0.7)
            
            ax1.yaxis.set_major_formatter(FuncFormatter(self._percentage_formatter))
            ax1.set_xticks([])
            ax1.set_title("Trade PnL")
            ax1.set_ylabel("PnL")

        # Net value curve
        if plot_type in ["line", "all"]:

            ax2.grid(True, linestyle="-", alpha=0.1)
            ax2.hlines(y=1, xmin=net_value.index.min(), xmax=net_value.index.max(), linestyle="--", color="grey")

            y = net_value["Benchmark"]
            fill_region_up = np.where(y >= 1, y, 1)
            fill_region_down = np.where(y < 1, y, 1)
            ax2.fill_between(x=net_value.index, y1=1, y2=fill_region_up, color="lightcoral", alpha=0.4)
            ax2.fill_between(x=net_value.index, y1=1, y2=fill_region_down, color="lightblue", alpha=0.4)
            
            if dynamic:
                
                line1, = ax2.plot([], [], linewidth=1.5, color="#FEE0D2", label="Account")
                line2, = ax2.plot([], [], linewidth=1.5, color="grey", label="Benchmark")

                dates_list = []
                account_list = []
                benchmark_list = []

                def init():
                    y_min = min(min(net_value["Account"]), min(net_value["Benchmark"]))
                    y_max = max(max(net_value["Account"]), max(net_value["Benchmark"]))
                    ax2.set_ylim(y_min, y_max)
                    return line1, line2

                def update(frame):
                    # Append new data points
                    dates_list.append(net_value.index[frame])
                    account_list.append(net_value["Account"].iloc[:frame+1].iloc[-1])
                    benchmark_list.append(net_value["Benchmark"].iloc[:frame+1].iloc[-1])
                    # Update lines
                    line1.set_data(dates_list, account_list)
                    line2.set_data(dates_list, benchmark_list)
                    return line1, line2

                # Create animation
                ani = FuncAnimation(fig, update, frames=len(net_value.index), init_func=init, blit=True, interval=10, repeat=False)
            else:
                ax2.plot(net_value["Account"], linewidth=1.5, color="#FEE0D2", label="Account")
                ax2.plot(net_value["Benchmark"], linewidth=1.5, color="grey", label="Benchmark")
            
            ax2.set_title("Portfolio Performance")
            ax2.set_xlabel("Trade Date")
            ax2.set_ylabel("Net Value")
            ax2.legend(loc="upper left", fontsize=12)

        # fig.tight_layout()
        plt.show()


    def return_dist_by_period(self, portolio_return, period="month", projection="2d", mode="light"):

        load_style(mode)

        daily_return = pd.DataFrame(portolio_return.Account)

        if projection == "2d":
            if period == "month":
                daily_return["label"] = daily_return.index.month
            if period == "year":
                daily_return["label"] = daily_return.index.year

            joyplot(daily_return, by="label", column="Account", ylim="own", 
                    figsize=(8, 6), colormap=plt.cm.twilight_shifted, alpha=0.5)  # cubehelix
            
            plt.xlabel("Returns", fontsize=12)
        
        if projection == "3d":
            if period == "month":
                daily_return["label"] = daily_return.index.month
                labels = range(1, 13)
                period_label = "Month"
            if period == "year":
                daily_return["label"] = daily_return.index.year
                labels = sorted(daily_return["label"].unique())
                period_label = "Year"

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")

            if mode == "dark":
                ax.xaxis.pane.set_facecolor("#252526")
                ax.yaxis.pane.set_facecolor("#252526")
                ax.zaxis.pane.set_facecolor("#252526")
                ax.xaxis._axinfo['grid'].update(color=(1, 1, 1, 0.1))
                ax.yaxis._axinfo['grid'].update(color=(1, 1, 1, 0.1))
                ax.zaxis._axinfo['grid'].update(color=(1, 1, 1, 0.1))

            x_range = np.linspace(daily_return.Account.min(), daily_return.Account.max(), 200).squeeze()
            colors = plt.cm.twilight_shifted(np.linspace(0, 1, len(labels)))
            for idx, (label, color) in enumerate(zip(labels, colors)):
                period_data = daily_return[daily_return.label == label]["Account"]
                if len(period_data) > 1:
                    kde = gaussian_kde(period_data)
                    density = kde(x_range)
                    ax.plot(x_range, np.ones_like(x_range) * idx, density, color=color, label=f"{period_label} {label}")
                    verts = [[x_range[i], idx, density[i]] for i in range(len(x_range))]
                    verts = [[x_range[0], idx, 0]] + verts + [[x_range[-1], idx, 0]]
                    ax.add_collection3d(Poly3DCollection([verts], facecolor=color, alpha=0.3))
            
            ax.view_init(elev=30, azim=45)
            ax.set_xlabel("Returns")
            ax.set_ylabel(period_label)
            ax.set_zlabel("Density")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
        
        plt.title(f"Daily Return Distribution by {period}", fontsize=15, pad=20)

        # plt.tight_layout()
        plt.show()


    def return_dist_with_benchmark(self, port_ret, hist=False, mode="light"):
        
        load_style(mode)
        plt.figure(figsize=(8, 5))

        # The easy version
        # sns.kdeplot(port_ret, shade=True, alpha=0.2, legend=True, edgecolor="none")

        colors = ["#7a8e93", "#d1c6bf"]
        for i, ret in enumerate(port_ret.columns):
            returns = port_ret[ret]
            if hist:
                sns.histplot(returns, kde=True, color=colors[i], line_kws={"linewidth": 2, "alpha": 0.5}, stat="density", label=ret, alpha=0.5, edgecolor="none")
            else:
                sns.kdeplot(returns, shade=True, label=ret, color=colors[i], alpha=0.8, legend=True, edgecolor="none")
        
        plt.legend(frameon=False)
        plt.title("Daily Return Distribution", fontsize=15, pad=20)
        plt.xlabel("Returns", fontsize=12)
        plt.ylabel("Density", fontsize=12)

        plt.show()


    def create_kde_plot(self, port_ret, section=None, marginal_hist=True, mode="light"):

        cmap = customize_colormap(["#b1bfbf", "#ded0ad", "#989c85"], "custom")
        
        load_style(mode)

        plt.figure()
        if section is None:
            g = sns.jointplot(
                    data=port_ret, x="Account", y="Benchmark", 
                    kind="kde", cmap=cmap, shade=True, zorder=0, n_levels=10, alpha=0.4, 
                    space=0, fill=True, ratio=8, height=6, marginal_kws={"color": "#7a8e93", "alpha": 0.5, "edgecolor": "none"})
            g.plot_joint(sns.regplot, scatter=False, line_kws={"color": "#695351", "alpha": 0.2})
            if marginal_hist:
                g.plot_marginals(sns.histplot, stat="density", alpha=0.5, color="#d1c6bf")

        else:
            total_length = len(port_ret)
            # the length of each section
            first_third = total_length // 3
            second_third = 2 * total_length // 3
            # set labels
            port_ret["label"] = 3
            port_ret.iloc[:first_third, port_ret.columns.get_loc("label")] = 1
            port_ret.iloc[first_third:second_third, port_ret.columns.get_loc("label")] = 2
            # TODO: year mapping
            # port_ret["label"] = port_ret.index.year.map({2018: 1, 2019: 2, 2020: 3})

            g = sns.jointplot(data=port_ret, x="Account", y="Benchmark", 
                              hue="label", kind="kde", zorder=0, levels=10, alpha=0.5, 
                              fill=True, space=0)
            if marginal_hist:
                g.plot_marginals(sns.histplot, stat="density", alpha=0.5)
        
        g.figure.suptitle("Strategy Returns KDE Plot", y=1.05, fontsize=15)
        plt.xlabel("Account", fontsize=12)
        plt.ylabel("Benchmark", fontsize=12)
        plt.show()
    

    def kde_2D(self, port_ret_1, port_ret_2, mode="light"):

        load_style(mode)

        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(4, 5, height_ratios=[1.5, 3, 3, 3], width_ratios=[3, 3, 3, 1.5, 1.5], figure=fig)

        # main kde plot
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        sns.kdeplot(
            data=port_ret_1, x="Account", y="Benchmark", 
            fill=True, alpha=0.6, cmap="Blues", cbar=False, thresh=0.05, n_levels=15, zorder=0, 
        )
        sns.kdeplot(
            data=port_ret_2, x="Account", y="Benchmark", 
            fill=True, alpha=0.4, cmap="Reds", cbar=False, thresh=0.05, n_levels=15, zorder=0, 
        )

        ax_main.set_xlabel("Account")
        ax_main.set_ylabel("Benchmark")

        # top histogram plot
        ax_top = fig.add_subplot(gs[0, 0:3])

        sns.histplot(data=port_ret_1, x="Account", stat="density", 
                    kde=False, color="#7DA6C6", alpha=0.5, ax=ax_top, 
                    legend=False, orientation="horizontal")
        sns.histplot(data=port_ret_2, x="Account", stat="density", 
                    kde=False, color="#E68B81", alpha=0.5, ax=ax_top, 
                    legend=False, orientation="horizontal")
        # "#82433c", "#adbbbe"
        ax_top.set_ylabel("Density")
        ax_top.set_xlabel("")
        ax_top.tick_params(labelbottom=True)

        # right histogram plot
        ax_right = fig.add_subplot(gs[1:4, 4])

        sns.histplot(data=port_ret_1, y="Benchmark", stat="density", 
                    kde=False, color="#7DA6C6", alpha=0.5, ax=ax_right, 
                    legend=False, orientation="horizontal")
        sns.histplot(data=port_ret_2, y="Benchmark", stat="density", 
                    kde=False, color="#E68B81", alpha=0.5, ax=ax_right, 
                    legend=False, orientation="horizontal")

        ax_right.set_xlabel("Density")
        ax_right.set_ylabel("")
        ax_right.tick_params(labelleft=True)

        # colorbar
        reds = plt.cm.Reds(np.linspace(0, 1, 256))
        reds[:, -1] = 0.5  # set alpha
        cmap_reds = plt.cm.colors.ListedColormap(reds)

        blues = plt.cm.Blues(np.linspace(0, 1, 256))
        blues[:, -1] = 0.5
        cmap_blues = plt.cm.colors.ListedColormap(blues)

        minmax1 = [port_ret_1.values.min(), port_ret_1.values.max()]
        minmax2 = [port_ret_2.values.min(), port_ret_2.values.max()]

        sm1 = plt.cm.ScalarMappable(cmap=cmap_reds, norm=plt.Normalize(vmin=minmax1[0], vmax=minmax1[1]))
        sm2 = plt.cm.ScalarMappable(cmap=cmap_blues, norm=plt.Normalize(vmin=minmax2[0], vmax=minmax2[1]))

        sm1.set_array([])
        sm2.set_array([])

        cax1 = fig.add_axes([0.75, 0.51, 0.01, 0.3])
        cax2 = fig.add_axes([0.75, 0.11, 0.01, 0.3])
        cbar1 = plt.colorbar(sm1, cax=cax1)
        cbar2 = plt.colorbar(sm2, cax=cax2)

        cbar1.ax.tick_params(labelsize=8)
        cbar2.ax.tick_params(labelsize=8)

        for cbar, minmax in zip([cbar1, cbar2], [minmax1, minmax2]):
            ticks = np.linspace(minmax[0], minmax[1], 6)   # create 6 equally weighted ticks
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])

        # cbar1.set_label("Red Scale", rotation=270, labelpad=15)
        # cbar2.set_label("Blue Scale", rotation=270, labelpad=15)

        plt.tight_layout()
        plt.show()


    def returns_heatmap(self, port_ret, cmap="RdBu_r", colorbar=True, alpha=1, mode="light"):
        """
        Create the heatmap of daily returns.

        Parameters:
        -----------
        port_ret: pd.DataFrame
            The portfolio returns, with columns "Account" and "Benchmark".

        cmap: str
            The colormap, default "RdBu_r", optional "RdYlGn_r".
        
        colorbar: bool
            Control whether to show the color bar.
        
        alpha: Union[int, float]
            Control the transparency of the plot.
        """

        load_style(mode)
        calplot.calplot(port_ret["Account"], cmap=cmap, colorbar=colorbar, alpha=alpha, 
                        yearlabel_kws = {"fontsize": 15, "color": "black"})
        plt.show()
    

    def industry_cashflow_chord(self):

        raise NotImplementedError


    
    # NOTE: The following methods are designed for batch parameter tuning.
    def create_tuning_plot(self, daily_return, mode="light"):
        
        load_style(mode)
        if mode == "light":
            cmap = plt.cm.viridis
        elif mode == "dark":
            cmap = customize_colormap(["lightgrey", "#98A3B8", "#BCC2B8", "#D1BDAF", "#C4AAB0", "#EFD3D9"], "")

        daily_return = daily_return.fillna(0)
        net_value = (1 + daily_return).cumprod()

        x = np.arange(len(net_value.columns)) + 1
        y = net_value.index
        X, Y = np.meshgrid(x, mdates.date2num(y))
        X = X.T
        Y = Y.T
        Z = net_value.values.T

        if len(net_value) <= 252*2:
            freq = "M"
        else:
            freq = "Y"
        
        date_index = net_value.index.searchsorted(net_value.resample(freq).last().index)[:-1]
        date_index = np.insert(date_index, 0, 0)
        date_index = np.insert(date_index, len(date_index), len(net_value.index)-1)

        # plt.rcParams["font.family"] = ["Times New Roman"]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        if mode == "dark":
            ax.xaxis.pane.set_facecolor("#252526")
            ax.yaxis.pane.set_facecolor("#252526")
            ax.zaxis.pane.set_facecolor("#252526")
            ax.xaxis._axinfo['grid'].update(color=(1, 1, 1, 0.1))
            ax.yaxis._axinfo['grid'].update(color=(1, 1, 1, 0.1))
            ax.zaxis._axinfo['grid'].update(color=(1, 1, 1, 0.1))

        for i in range(Z.shape[0]):
            
            ax.plot(X[i], Y[i], Z[i], alpha=0)
            # ax.plot(X[i], Y[i], Z[i], color=plt.cm.viridis(i/len(x)), alpha=0.3, label=f"batch {i}")
            # ax.plot(X[i], Y[i], np.zeros_like(Z[i]), alpha=0)

            verts = [[X[i, j], Y[i, j], 0] for j in range(len(y))] + [[X[i, j], Y[i, j], Z[i, j]] for j in range(len(y)-1,-1,-1)]
            ax.add_collection3d(Poly3DCollection([verts], color=cmap(i/len(x)), alpha=0.3))
            
            for j in date_index:
                ax.text(X[i, j], Y[i, j], Z[i, j] + 0.03, f"{Z[i, j]:.2f}", fontsize=8, color="black", ha="center")
            
        for j in date_index:
            ax.plot(X[:, j], Y[:, j], Z[:, j], "--", linewidth=1, color="gray")

        legend_elements = [Rectangle((0, 0), 1, 1, facecolor=cmap(i/len(x)), edgecolor="none", alpha=0.3) for i in range(len(x))]
        legend_texts = net_value.columns

        ax.set_xticks(x, legend_texts)
        # ax.set_xticklabels(legend_texts, fontsize=12)
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
        # ax.yaxis.set_major_locator(mdates.MonthLocator())
        ax.yaxis.set_tick_params(labelsize=10)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=-30, ha="left")
        ax.zaxis._axinfo["juggled"] = (1, 2, 0)

        # ax.set_xlabel("Batch No.")
        # ax.set_ylabel("Trade Date")
        # ax.set_zlabel("Net Value")
        ax.set_zlim([0, Z.max()])
        title_color = "ghostwhite" if mode == "dark" else "black"
        ax.set_title("Strategy net value", fontsize=16, fontweight="bold", color=title_color, fontstyle="italic", verticalalignment="top")
        ax.legend(legend_elements, legend_texts)

        ax.view_init(elev=20, azim=-45)

        plt.tight_layout()
        plt.show()
    

    def returns_dist_plot(self, daily_returns, type="ridge", mode="light"):

        daily_returns.index.name = "trade_date"
        daily_returns.columns = np.arange(daily_returns.shape[1]) + 1
        data_melted = daily_returns.melt(ignore_index=False, var_name="batch", value_name="returns")
        data_melted = data_melted.reset_index()

        load_style(mode)
        fig = plt.figure(figsize=(10, 6))

        if type == "overlap":
            sns.kdeplot(
                data=data_melted, x="returns", hue="batch",
                fill=True, common_norm=False, palette="crest",
                alpha=0.2, linewidth=0, 
                )
            # sns.kdeplot(daily_return, shade=True, alpha=0.2, legend=True, edgecolor="none")

        if type == "ridge":
            data_array = data_melted["returns"].dropna()
            min_cut = np.percentile(data_array, 1)
            max_cut = np.percentile(data_array, 99)

            joyplot(data_melted, by="batch", column="returns", ylim="own", 
                    figsize=(8, 6), x_range=(min_cut, max_cut), colormap=plt.cm.RdBu_r, alpha=0.5)
        
        plt.title("Daily Return Distribution by Batch", fontsize=15, pad=20)
        plt.xlabel("Returns", fontsize=12)
        plt.show()


perf_plot = STGPlot()


class InteractivePlot:
    def __init__(self):
        """Initialize plot object."""
        load_style("dark")
        plt.ion()  # turn on interactive mode
        
        # create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(15, 6))
        self._setup_axes()
        
        # create line and fill region
        self.line1, self.line2 = self._create_lines()
        self.fill1, self.fill2 = self._create_fills()
    
    def _setup_axes(self):
        """Set up axes attributes."""
        self.ax.set_title("Portfolio Performance")
        self.ax.set_xlabel("Trade Date")
        self.ax.set_ylabel("Net Value")
        self.ax.grid(True, linestyle="-", alpha=0.1)
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        # plt.gcf().autofmt_xdate()
    
    def _create_lines(self):
        """Create line object."""
        line1, = self.ax.plot([], [], linewidth=1.5, color="#FEE0D2", label="Account")
        line2, = self.ax.plot([], [], linewidth=1.5, color="grey", label="Benchmark")
        return line1, line2
    
    def _create_fills(self):
        """Create fill region object."""
        fill1 = self.ax.fill_between([], [], 1, color="lightcoral", alpha=0.4)
        fill2 = self.ax.fill_between([], [], 1, color="lightblue", alpha=0.4)
        return fill1, fill2
    
    def update(self, dates, account_rets, market_rets, is_first_update=False):
        """Update figure data."""
        # compute cumulative returns
        account_values = (1 + pd.Series(account_rets)).cumprod().values
        market_values = (1 + pd.Series(market_rets)).cumprod().values

        # update line data
        self.line1.set_data(dates, account_values)
        self.line2.set_data(dates, market_values)

        # remove old fills
        self.fill1.remove()
        self.fill2.remove()
        
        # update fills
        fill_region_up = np.where(market_values >= 1, market_values, 1)
        fill_region_down = np.where(market_values < 1, market_values, 1)
        self.fill1 = self.ax.fill_between(dates, 1, fill_region_up, color="lightcoral", alpha=0.4)
        self.fill2 = self.ax.fill_between(dates, 1, fill_region_down, color="lightblue", alpha=0.4)

        # adjust axes
        self.ax.relim()
        self.ax.autoscale_view()
        
        # add legend for the first update
        if is_first_update:
            self.ax.legend(loc="upper left", fontsize=12)

        # flush figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # time.sleep(0.1)
    
    def show(self):
        """Show plot."""
        plt.ioff()
        plt.show()


class AlphaPlot:

    def __init__(self, config: "FrozenConfig", alpha=None, prices=None, mode="light"):
        
        load_style(mode, module="factor")
        
        self._alpha_raw = alpha
        self._prices_raw = prices
        self._alpha = None
        self._prices = None
        self._config = config
    
    @property
    def prices(self):
        """Lazy load price data"""
        if self._prices is None:
            if self._prices_raw is None:
                raise ValueError("Prices must be provided to analyze factor")
            self._prices = self._prices_raw
        return self._prices
    
    @property
    def alpha(self):
        """Lazy load factor data"""
        if self._alpha is None:
            if self._alpha_raw is None:
                raise ValueError("Alpha must be provided")
            if isinstance(self._alpha_raw, _lazy_import_factor()):
                self._alpha = self._alpha_raw.data[self._config.start_date:]
            else:
                self._alpha = self._alpha_raw[self._config.start_date:]
        return self._alpha
    
    def set_alpha(self, alpha):
        """Set factor data"""
        self._alpha_raw = alpha
        self._alpha = None  # reset cache
    
    def set_prices(self, prices):
        """Set prices"""
        self._prices_raw = prices
        self._prices = None  # reset cache
    

    def period_ic_ts(self, period):

        period_rets = self.prices.pct_change(period).shift(-period)
        period_rets = period_rets[self._config.start_date:]
        period_ic = self.alpha.corrwith(period_rets, axis=1, method="spearman")
        ic_mean = period_ic.mean()
        ic_std = period_ic.std()
        
        fig = plt.figure(figsize=(15, 5))

        plt.plot(period_ic, color="#90A5A7", alpha=0.6)
        plt.xticks(rotation=30)
        plt.xlabel("Trade Date")
        plt.ylabel("IC")
        plt.title(f"{period}D Period Forward Return Information Coefficient (IC)", pad=15)

        plt.annotate(f"Mean: {ic_mean:.3f}\nStd: {ic_std:.3f}", 
                    xy=(0.05, 0.95), 
                    xycoords="axes fraction", 
                    bbox=dict(boxstyle="round, pad=0.6", fc="#FAE9DC", ec="#FAE9DC", alpha=0.2), 
                    va="top", 
                    ha="left", 
                    fontsize=12, 
                    alpha=0.7)

        plt.grid(False)
        plt.show()
    

    def period_ic_bar(self, total_period):

        ic = np.zeros(total_period)
        for i, period in enumerate(range(1, total_period+1)):
            period_rets = self.prices.pct_change(period).shift(-period)
            period_rets = period_rets[self._config.start_date:]
            ic[i] = self.alpha.corrwith(period_rets, axis=1, method="spearman").mean()
        
        periods = [f"{period}D" for period in range(1, total_period+1)]
        crest_cmap = customize_colormap(["#1C3F5F", "#1D5F7C", "#2C7C95", "#4E9BAA", "#7FBCBB", "#B2DBCD"], "crest")
        colors = crest_cmap(np.linspace(0, 1, total_period)) # plt.cm.coolwarm(np.linspace(0, 1, total_period))

        fig = plt.figure(figsize=(10, 6))
        plt.bar(periods, ic, color=colors, alpha=0.5)
        plt.grid(False)
        plt.title("IC Trend w.r.t. Forward Periods")
        plt.xlabel("Periods")
        plt.ylabel("IC")

        # sm = plt.cm.ScalarMappable(cmap=crest_cmap, norm=plt.Normalize(vmin=ic.min(), vmax=ic.max()))
        # sm.set_array([])
        # cbar = plt.colorbar(sm, alpha=0.5, shrink=0.8)
        # cbar.set_label("IC Scale")

        plt.grid(False)
        plt.show()


    def factor_net_value(self, factor: "Factor" = None):
        """
        Calculate factor net value by constructing a long-short portfolio based on factor values.
        
        This function creates a factor-mimicking portfolio by:
        1. Standardizing factor values cross-sectionally at each time point
        2. Converting standardized values to portfolio weights
        3. Computing portfolio returns using the weights and stock returns
        4. Calculating cumulative net value over time
        
        Returns:
        --------
        pd.Series
            Factor net value series indexed by trade dates
        
        Notes:
        ------
        - Factor values are standardized cross-sectionally at each time point
        - Weights are calculated as normalized factor values (sum of absolute weights = 1)
        - Missing values are filled with 0
        """
        if factor is not None:
            factor_data = factor.data
        else:
            factor_data = self.alpha
        returns_data = self.prices.pct_change(1)

        # Align data index
        common_dates = factor_data.index.intersection(returns_data.index)
        factor_aligned = factor_data.loc[common_dates]
        returns_aligned = returns_data.loc[common_dates]
        
        # Remove missing values
        factor_clean = factor_aligned.fillna(0)
        returns_clean = returns_aligned.fillna(0)
        
        # Periodically normalize factors
        factor_normalized = factor_clean.sub(factor_clean.mean(axis=1), axis=0).div(factor_clean.std(axis=1), axis=0)
        
        # Calculate weights
        weights = factor_normalized.div(factor_normalized.abs().sum(axis=1), axis=0)
        
        # Calculate portfolio returns
        portfolio_returns = (weights * returns_clean).sum(axis=1)
        
        # Calculate net value
        net_value = (1 + portfolio_returns).cumprod()
        
        return net_value


    def factor_net_value_by_batch(self, factors: Dict[str, "Factor"]):
        """
        Calculate net value for multiple factors in batch processing.
        
        Parameters:
        -----------
        factors : Dict[str, Factor]
            Dictionary containing factor objects with keys as factor names
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with factor names as columns and net value series as rows
        
        Notes:
        ------
        - Each factor net value is calculated using factor_net_value() function
        - Returns a dictionary with factor names as keys and net value series as values
        """
        factor_net_values = {}
        
        for factor_name, factor_obj in factors.items():
            net_value = self.factor_net_value(factor_obj)
            factor_net_values[factor_name] = net_value
        
        return pd.concat(factor_net_values, axis=1)
    

    def factor_layers(self, log_ret=False, demeaned=False, long_short=False, n_groups=5):

        cum_rets = self._factor_quantile_analysis(log_ret, demeaned, long_short, n_groups)
        
        plt.figure(figsize=(12, 6))
        # 1. matplotlib color mapping, such as "viridis", "plasma", "inferno", "magma"
        cmap = plt.cm.get_cmap("coolwarm")
        # 2. seaborn palette, such as "husl", "muted", "deep", "colorblind"
        # palette = sns.color_palette("coolwarm", len(cum_rets.columns))
        # palette = sns.cubehelix_palette(len(cum_rets.columns), start=.5, rot=-.75)
        # 3. assigned color list
        # custom_colors = ["#C3D5D0", "#C2C6AD", "#EFE3B2", "#D9A076"]
        # YZMC: ["#C3D5D0", "#C2C6AD", "#EFE3B2", "#D9A076"]
        # CHWJ: ["#F1F1F1", "#D1E2EA", "#D8CAE0", "#A8A9C0"]
        # XHRX: ["#FFE5E5", "#FFCCCE", "#C6E1FF", "#DEEDFF"]

        colors = cmap(np.linspace(0, 1, len(cum_rets.columns)))

        for i, column in enumerate(cum_rets.columns):
            plt.plot(cum_rets.index, cum_rets[column], label=column, color=colors[i], alpha=0.5)
        
        plt.title("1D Forward Cumulative Returns of Factor Quantiles")
        plt.xlabel("Trade Date")
        plt.ylabel("Cumulative Return")
        plt.legend(title="Groups")

        plt.grid(False)
        plt.show()


    def _factor_quantile_analysis(self, log_ret=False, demeaned=False, long_short=False, n_groups=5):

        factor_mat_raw = self.alpha.shift(1)
        returns_mat_raw = self.prices.pct_change(1)

        factor_mat, returns_mat = (
            self._validate_and_transform_factor_returns(factor_mat_raw, returns_mat_raw)
            )

        if log_ret:
            returns_mat = np.log(1 + returns_mat)

        quantile_returns = []

        # factor layer by date
        for date in factor_mat.index:
            factors = factor_mat.loc[date]
            returns = returns_mat.loc[date]
            
            # drop NAs
            valid = ~(factors.isna() | returns.isna())
            factors = factors[valid]
            returns = returns[valid]

            # fill with zero if valid factor is empty
            if len(factors) == 0:
                quantile_returns.append(pd.Series(np.zeros(n_groups), name=date))
                continue
            
            # demean returns
            if demeaned:
                returns = returns - returns.mean()

            # sort and group by factor values
            labels = pd.qcut(factors, q=n_groups, labels=False, duplicates="drop")
            
            # average returns by group
            group_returns = returns.groupby(labels).mean()
            quantile_returns.append(group_returns)
        
        # cumulative returns by group
        quantile_returns = pd.DataFrame(quantile_returns, index=factor_mat.index)
        quantile_returns.columns = np.arange(1, n_groups+1)
        cumulative_returns = (1 + quantile_returns).cumprod()

        if long_short:
            long_short_return = quantile_returns.iloc[:, -1] - quantile_returns.iloc[:, 0]
            cum_ls_return = (1 + long_short_return).cumprod()
            cum_ls_return = pd.Series(cum_ls_return, name="long-short")
            cumulative_returns = pd.concat([cumulative_returns, cum_ls_return], axis=1)

        return cumulative_returns
    

    def _validate_and_transform_factor_returns(self, factor_mat, returns_mat, verbose=True):
        """
        Validate the consistency between factor matrix and returns matrix, 
        and handle special cases.
        
        Args:
            factor_mat (DataFrame): Factor matrix with date index.
            returns_mat (DataFrame): Returns matrix with date index.
            verbose (bool): Whether to print warning messages.
        
        Returns:
            tuple: (Validated factor_mat, Validated returns_mat)
        """

        factor_mat = factor_mat.copy()
        returns_mat = returns_mat.copy()
        
        # 1. Check for rows in factor_mat that are all NaN
        all_na_dates = factor_mat.index[factor_mat.isna().all(axis=1)]
        
        # 2. Find dates that are all NaN in factor_mat but missing in returns_mat
        missing_in_returns = [date for date in all_na_dates if date not in returns_mat.index]
        
        # 3. Remove rows for these dates
        if missing_in_returns:
            factor_mat = factor_mat.drop(missing_in_returns)
            if verbose:
                warnings.warn(f"Dropped {len(missing_in_returns)} rows because these dates were all NaN in factor_mat but missing in returns_mat.")
        
        # 4. Check if remaining indices are consistent
        common_index = factor_mat.index.intersection(returns_mat.index)
        if len(common_index) < len(factor_mat) or len(common_index) < len(returns_mat):
            factor_mat = factor_mat.loc[common_index]
            returns_mat = returns_mat.loc[common_index]
            if verbose:
                warnings.warn(f"Indices were inconsistent. Restricted both factor_mat and returns_mat to common dates: {len(common_index)} rows")
        
        return factor_mat, returns_mat


class FactorPlot:

    @staticmethod
    def plot_factor_price_relation(
        price_data: pd.DataFrame,
        factor_data: pd.Series,
        ticker: str,
        factor_name: str = "Factor",
        title: Optional[str] = None,
        height: int = 800,
        show_volume: bool = True,
        show_nontrading: bool = False,
        factor_color: str = "blue",
        candlestick_colors: Dict[str, str] = None,
        dark_theme: bool = False
    ) -> go.Figure:
        """
        Plot the relationship between factor values and stock price using candlestick and bar charts.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data with columns ['open', 'high', 'low', 'close', 'volume']
            Index should be DatetimeIndex
        factor_data : pd.Series
            Factor values with DatetimeIndex
        ticker : str
            Stock ticker symbol
        factor_name : str, default "Factor"
            Name of the factor for labeling
        title : str, optional
            Custom title for the plot
        height : int, default 800
            Height of the plot in pixels
        show_volume : bool, default True
            Whether to show volume subplot
        show_nontrading : bool, default False
            Whether to show non-trading days
        factor_color : str, default "blue"
            Color for factor bars
        candlestick_colors : Dict[str, str], optional
            Custom colors for candlestick chart
        dark_theme : bool, default False
            Theme for figure
        
        Returns:
        --------
        plotly.graph_objects.Figure
            The interactive plot figure
        """
        
        # Validate inputs
        if not isinstance(price_data.index, pd.DatetimeIndex):
            raise ValueError("price_data index must be DatetimeIndex")
        if not isinstance(factor_data.index, pd.DatetimeIndex):
            raise ValueError("factor_data index must be DatetimeIndex")
        
        required_price_cols = ["open", "high", "low", "close"]
        if not all(col in price_data.columns for col in required_price_cols):
            raise ValueError(f"price_data must contain columns: {required_price_cols}")
        
        # Align data by date
        common_dates = price_data.index.intersection(factor_data.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates found between price_data and factor_data")
        
        price_aligned = price_data.loc[common_dates].sort_index()
        factor_aligned = factor_data.loc[common_dates].sort_index()
        
        # Set default candlestick colors
        if candlestick_colors is None:
            candlestick_colors = {
                "increasing": "#26a69a",  # Teal Green
                "decreasing": "#ef5350"   # Red
            }
        
        # Create subplots
        if show_volume and "volume" in price_data.columns:
            subplot_titles = [
                f"{ticker} Price & {factor_name}",
                f"{factor_name} Values", 
                "Volume"
            ]
            specs = [
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}]
            ]
            rows = 3
            row_heights = [0.6, 0.25, 0.15]
        else:
            subplot_titles = [
                f"{ticker} Price & {factor_name}",
                f"{factor_name} Values"
            ]
            specs = [
                [{"secondary_y": True}],
                [{"secondary_y": False}]
            ]
            rows = 2
            row_heights = [0.7, 0.3]
        
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            specs=specs,
            row_heights=row_heights
        )
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=price_aligned.index,
            open=price_aligned["open"],
            high=price_aligned["high"],
            low=price_aligned["low"],
            close=price_aligned["close"],
            name="Price",
            increasing_line_color=candlestick_colors["increasing"],
            decreasing_line_color=candlestick_colors["decreasing"],
            increasing_fillcolor=candlestick_colors["increasing"],
            decreasing_fillcolor=candlestick_colors["decreasing"],
            line=dict(width=1),
        )
        fig.add_trace(candlestick, row=1, col=1)
        
        # Add factor bars in the second subplot
        factor_colors = [factor_color if val >= 0 else "red" for val in factor_aligned.values]
        
        factor_bars = go.Bar(
            x=factor_aligned.index,
            y=factor_aligned.values,
            name=factor_name,
            marker_color=factor_colors,
            opacity=0.7,
            text=[f"{val:.3f}" for val in factor_aligned.values],
            textposition="outside",
            textfont=dict(size=10)
        )
        fig.add_trace(factor_bars, row=2, col=1)
        
        # Add volume if requested
        if show_volume and "volume" in price_data.columns:
            volume_aligned = price_data.loc[common_dates, "volume"].sort_index()
            
            # Color volume bars based on price movement
            volume_colors = []
            for i, date in enumerate(volume_aligned.index):
                if i == 0:
                    color = "gray"
                else:
                    prev_close = price_aligned.iloc[i-1]["close"]
                    curr_close = price_aligned.iloc[i]["close"]
                    color = candlestick_colors["increasing"] if curr_close >= prev_close else candlestick_colors["decreasing"]
                volume_colors.append(color)
            
            volume_bars = go.Bar(
                x=volume_aligned.index,
                y=volume_aligned.values,
                name="Volume",
                marker_color=volume_colors,
                opacity=0.6
            )
            fig.add_trace(volume_bars, row=3, col=1)
        
        # Update layout
        if title is None:
            title = f"{ticker} - {factor_name} vs Price Analysis"
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            height=height,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode="x unified"
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text=factor_name, row=2, col=1)
        if show_volume and 'volume' in price_data.columns:
            fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        # Update x-axes
        fig.update_xaxes(title_text="Date", row=rows, col=1)
        
        # Add horizontal line at factor = 0
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

        # Remove non-trading days
        if not show_nontrading:
            start_date = price_aligned.index.min()
            end_date = price_aligned.index.max()
            dt_all = pd.date_range(start=start_date, end=end_date, freq="D")
            dt_breaks = [d for d in dt_all if not d in common_dates]
            fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

        # Additional customization
        if dark_theme:
            fig.update_layout(
                    template="plotly_dark",  # Dark theme
                    font=dict(family="Arial", size=12),
                    title_font_size=24
                )
        
        return fig


    @staticmethod
    def plot_factor_distribution(
        factor_data: pd.Series,
        factor_name: str = "Factor",
        bins: int = 50,
        show_stats: bool = True,
        color: str = "blue"
    ) -> go.Figure:
        """
        Plot factor distribution histogram with statistics.
        
        Parameters:
        -----------
        factor_data : pd.Series
            Factor values
        factor_name : str
            Name of the factor
        bins : int
            Number of histogram bins
        show_stats : bool
            Whether to show statistics on the plot
        color : str
            Color for the histogram
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        
        # Remove NaN values
        clean_data = factor_data.dropna()
        
        if len(clean_data) == 0:
            raise ValueError("No valid factor data found")
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=clean_data.values,
            nbinsx=bins,
            name=factor_name,
            marker_color=color,
            opacity=0.7
        ))
        
        # Add statistics if requested
        if show_stats:
            mean_val = clean_data.mean()
            std_val = clean_data.std()
            median_val = clean_data.median()
            
            # Add vertical lines for statistics
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                        annotation_text=f"Mean: {mean_val:.3f}")
            fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                        annotation_text=f"Median: {median_val:.3f}")
            
            # Add statistics text
            stats_text = (
                f"Statistics:<br>"
                f"Mean: {mean_val:.4f}<br>"
                f"Std: {std_val:.4f}<br>"
                f"Median: {median_val:.4f}<br>"
                f"Min: {clean_data.min():.4f}<br>"
                f"Max: {clean_data.max():.4f}<br>"
                f"Count: {len(clean_data)}"
            )
            
            fig.add_annotation(
                x=0.98, y=0.98,
                text=stats_text,
                xref="paper", yref="paper",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        
        fig.update_layout(
            title=f"{factor_name} Distribution",
            xaxis_title=factor_name,
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig


    @staticmethod
    def plot_factor_time_series(
        factor_data: pd.Series,
        factor_name: str = "Factor",
        show_rolling_mean: bool = True,
        rolling_window: int = 20,
        color: str = "blue"
    ) -> go.Figure:
        """
        Plot factor time series with optional rolling statistics.
        
        Parameters:
        -----------
        factor_data : pd.Series
            Factor values with DatetimeIndex
        factor_name : str
            Name of the factor
        show_rolling_mean : bool
            Whether to show rolling mean
        rolling_window : int
            Window for rolling statistics
        color : str
            Color for the line
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        
        if not isinstance(factor_data.index, pd.DatetimeIndex):
            raise ValueError("factor_data index must be DatetimeIndex")
        
        fig = go.Figure()
        
        # Add main time series
        fig.add_trace(go.Scatter(
            x=factor_data.index,
            y=factor_data.values,
            mode="lines",
            name=factor_name,
            line=dict(color=color)
        ))
        
        # Add rolling mean if requested
        if show_rolling_mean:
            rolling_mean = factor_data.rolling(window=rolling_window).mean()
            fig.add_trace(go.Scatter(
                x=rolling_mean.index,
                y=rolling_mean.values,
                mode="lines",
                name=f"{rolling_window}-day MA",
                line=dict(color='red', dash='dash')
            ))
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f"{factor_name} Time Series",
            xaxis_title="Date",
            yaxis_title=factor_name,
            hovermode="x unified"
        )
        
        return fig
