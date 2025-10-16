import os
import fire
import importlib

def import_class_from_file(file_name, class_name):

    module_name = ".".join(["demo", "strategy", file_name])
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    return cls

def workflow(file_name, plot_type="line", excel=False, html=False):

    FactorFactory = import_class_from_file(file_name, "FactorFactory")
    instance = FactorFactory()
    instance.calc()
    instance.run_backtest(plot_type=plot_type, excel=excel, html=html)


def run():
    fire.Fire(workflow)

if __name__ == "__main__":
    run()
    