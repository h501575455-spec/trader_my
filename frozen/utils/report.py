import os
import mpld3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import quantstats as qs
from io import StringIO
from itertools import chain
from typing import Union, Tuple, Sequence, TYPE_CHECKING
from tabulate import tabulate

from .metrics import StrategyMetric
from .helper import _lazy_import_factor
from ..factor.extensions import alphalens as al

if TYPE_CHECKING:
    from ..factor.expression.base import Factor

class Report:

    @staticmethod
    def print_evaluation_table(port_ret, win_rate, print_table=True):
        """
        Prints a table with various financial performance metrics 
        for a portfolio and its benchmark. 
        
        It provides a comprehensive overview of the portfolio's 
        performance relative to its benchmark, allowing for quick 
        assessment of risk-adjusted returns, market sensitivity, 
        and other key performance metrics.

        Example of the table:

        .. code-block:: text

        +-----------+-------------+---------------+--------+---------+----------+-----------+--------+------------+----------------+
        |   Account |   Benchmark |   Annual Rate |   Beta |   Alpha |   Sharpe |   Sortino |     IR |   Win Rate |   Max Drawdown |
        +===========+=============+===============+========+=========+==========+===========+========+============+================+
        |    3.0609 |      0.8451 |        0.2806 |  0.674 |  0.3052 |   1.0208 |    1.5695 | 0.0747 |     0.4725 |         0.3371 |
        +-----------+-------------+---------------+--------+---------+----------+-----------+--------+------------+----------------+
        """

        daily_return = port_ret["Account"]
        net_value = (1 + port_ret).cumprod()
        final_nv = net_value[-1:].iloc[0,]
        ar = StrategyMetric.annual_rate(daily_return)
        beta = StrategyMetric.Beta(port_ret)
        alpha = StrategyMetric.Alpha(port_ret)
        sharpe = StrategyMetric.Sharpe_Ratio(daily_return)
        sortino = StrategyMetric.Sortino_Ratio(daily_return)
        information = StrategyMetric.Information_Ratio(port_ret)
        max_drawdown = StrategyMetric.Max_Drawdown(net_value["Account"])

        table = {"Account": [round(final_nv["Account"], 4)],
                 "Benchmark": [round(final_nv["Benchmark"], 4)],
                 "Annual Rate": [round(ar, 4)],
                 "Beta": [round(beta, 4)],
                 "Alpha": [round(alpha, 4)],
                 "Sharpe": [round(sharpe, 4)],
                 "Sortino": [round(sortino, 4)],
                 "IR": [round(information, 4)],
                 "Win Rate": [round(win_rate, 4)],
                 "Max Drawdown": [round(max_drawdown, 4)]}
        
        if print_table:
            print(tabulate(table, headers="keys", tablefmt="grid"))

        return table
    

    @staticmethod
    def to_excel(df, file_path, type):
        """
        Saves a pandas DataFrame to an Excel spreadsheet.

        Parameters:
        -----------
        df: pd.DataFrame
            The DataFrame to save.
        file_path: str
            The path to save the Excel file.
        type: str
            The type of file to save.

            - "positions": Save simple positions of the instrument.
            - "trades": Save trades details of the instrument.
        """
        if type == "positions":
            EXCEL_PATH = "instrument orders.xlsx"
        elif type == "trades":
            EXCEL_PATH = "trade records.xlsx"
        df.to_excel(os.path.join(file_path, EXCEL_PATH))


    @staticmethod
    def to_quantstats(port_ret, file_path):
        """
        Output quantstats html tearsheet.

        Reference:
            https://github.com/ranaroussi/quantstats
        """
        plt.style.use("default")
        HTML_PATH = "quantstats-tearsheet.html"
        qs.reports.html(port_ret["Account"], 
                        benchmark=port_ret["Benchmark"], 
                        title="Strategy Performance", 
                        output=os.path.join(file_path, HTML_PATH))


    @staticmethod
    def report_factor_analysis(
            alpha: Union["Factor", pd.DataFrame], 
            prices: pd.DataFrame, 
            quantiles: Union[int, Sequence[float]] = 5, 
            periods: Sequence[int] = (1, 5, 20),
            type: str = "all",
            render: str = "pic", 
            supress: bool = True, 
            output_dir: str = ""
        ):

        if isinstance(alpha, _lazy_import_factor()):
            alpha = alpha.data.stack().reset_index()
        else:
            alpha = alpha.stack().reset_index()
        
        alpha.columns = ["date", "ticker", "Alpha"]
        alpha.set_index(["date", "ticker"], inplace=True)

        data = al.utils.get_clean_factor_and_forward_returns(alpha, prices, quantiles=quantiles, periods=periods)

        if supress:
            # Use non-interactive backend
            matplotlib.use("Agg")
        
        if type == "returns":
            al.tears.create_returns_tear_sheet(data, render=render, output_dir=output_dir)
        elif type == "summary":
            al.tears.create_summary_tear_sheet(data)
        elif type == "turnover":
            al.tears.create_turnover_tear_sheet(data, render=render, output_dir=output_dir)
        elif type == "information":
            al.tears.create_information_tear_sheet(data, render=render, output_dir=output_dir)
        elif type == "all":
            al.tears.create_full_tear_sheet(data, render=render, output_dir=output_dir)
    

    @staticmethod
    def report_batch_results(results: Tuple, parallel=False) -> Tuple:
        """
        Process backtesting results of all batches at one time.
        
        Parameters:
        -----------
        results: list of tuple
            Tuple with elements (details, port_ret, ind).
        
        parallel: boolean
            Bool value to control the process mode.
        
        Returns:
        --------
        Tuple
            The processed result with elements (port_rets, inds).
        """
        if parallel:
            # Reduce, maintaining order across different n_jobs
            results = list(chain.from_iterable(results))
            param_col = pd.DataFrame(results[0:len(results):3])
            ind_col = pd.concat(results[2:len(results):3], ignore_index=True)
            port_rets = pd.concat([res["Account"] for res in results[1:len(results):3]], axis=1)
            # Formalize output
            port_rets.columns = ["batch " + str(num+1) for num in range(len(param_col))]
            inds = pd.concat([ind_col, param_col], axis=1)
            inds.set_index("batch", inplace=True)
        else:
            port_rets = pd.DataFrame()
            inds = pd.DataFrame()
            batch_serial = []
            # Loop over results and process batch by batch
            for details, port_ret, ind in results:
                # Process metrics
                ind["params"] = [details["params"]]
                inds = pd.concat([inds, ind], axis=0)
                # Process portfolio returns
                port_rets = pd.concat([port_rets, port_ret["Account"]], axis=1)
                batch_serial.append(details["batch"])
            # Formalize index and columns
            port_rets.index = pd.to_datetime(port_rets.index)
            batch_list = ["batch " + str(num) for num in batch_serial]
            port_rets.columns = batch_list
            inds.index = batch_list
        
        return port_rets, inds


report = Report()
