import optuna
from functools import partial
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

from frozen.utils.metrics import FactorMetric


class TuneBase(ABC):
     
    @abstractmethod
    def autotune(self,):
        raise NotImplementedError
    
    def _preprocess_param_space(self, param_space: dict) -> dict:
        """Preprocess parameter space and generate actual sampling space."""
        sampling_space = {}
        
        # find all independent parameters and dependent parameters
        independent_params = {}
        dependent_params = {}
        
        for name, config in param_space.items():
            if config["type"] == "tuple":
                sampling_space[f"{name}_a"] = {
                    "type": "int",
                    "min": config["lower_min"],
                    "max": config["lower_max"],
                    "_original_name": name,
                    "_is_tuple": True,
                    "_position": 0
                }
                sampling_space[f"{name}_b"] = {
                    "type": "int",
                    "min": config["upper_min"],
                    "max": config["upper_max"],
                    "_original_name": name,
                    "_is_tuple": True,
                    "_position": 1
                }
            elif "dependent" in config:
                dependent_params[name] = config
            else:
                independent_params[name] = config
                
        # process independent variables
        for name, config in independent_params.items():
            sampling_space[name] = config
            
        # process dependent variables
        for name, config in dependent_params.items():
            param_type = config["type"]
            dependent = config["dependent"]
            conditions = config["conditions"]
            # create corresponding parameters for every possible dependency
            for parent_value, choices in conditions.items():
                if choices is not None:  # create new fields only for options not None
                    param_name = f"{name}_{parent_value}"
                    # For categorical parameters
                    if param_type == "categorical":
                        sampling_space[param_name] = {
                            "type": "categorical",
                            "choice": choices,
                            "_original_name": name,  # save parameter original name
                            "_parent_name": dependent,  # save parent parameter name
                            "_parent_value": parent_value  # save parent parameter value
                        }
                    # For int/float parameters
                    elif param_type in ["int", "float"]:
                        sampling_space[param_name] = {
                            "type": param_type,
                            "min": choices["min"],
                            "max": choices["max"],
                            "step": choices.get("step", 1 if param_type == "int" else None),
                            "log": choices.get("log", False),
                            "_original_name": name,
                            "_parent_name": dependent,
                            "_parent_value": parent_value
                        }
        
        return sampling_space

    def suggest_params(self, trial: optuna.Trial, param_space: dict) -> Dict[str, Any]:
        """Generate optuna parameters from given parameter space."""
        # obtain sampling space
        sampling_space = self._preprocess_param_space(param_space)
        suggested_params = {}
        
        # First, sample all independent parameters (non-dependent and non-tuple)
        for param_name, param_config in sampling_space.items():
            if "_original_name" not in param_config and "_is_tuple" not in param_config:
                if param_config["type"] == "int":
                    value = trial.suggest_int(
                        param_name,
                        param_config["min"],
                        param_config["max"],
                        step=param_config.get("step", 1),
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "float":
                    value = trial.suggest_float(
                        param_name,
                        param_config["min"],
                        param_config["max"],
                        step=param_config.get("step", None),
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "categorical":
                    value = trial.suggest_categorical(
                        param_name,
                        param_config["choice"]
                    )
                suggested_params[param_name] = value
        
        # Then, sample independent tuple parameters
        tuple_params = {}
        for param_name, param_config in sampling_space.items():
            if "_is_tuple" in param_config and "_parent_name" not in param_config:
                original_name = param_config["_original_name"]
                position = param_config["_position"]
                if original_name not in tuple_params:
                    tuple_params[original_name] = {}
                if position == 0:
                    # Sample first element
                    value = trial.suggest_int(
                        param_name,
                        param_config["min"],
                        param_config["max"]
                    )
                    tuple_params[original_name]["a"] = value
                else:
                    # Sample second element, ensuring it's not less than the first
                    a_value = tuple_params[original_name]["a"]
                    value = trial.suggest_int(
                        param_name,
                        max(param_config["min"], a_value),
                        param_config["max"]
                    )
                    tuple_params[original_name]["b"] = value
        
        # Add independent tuple parameters to result
        for name, values in tuple_params.items():
            suggested_params[name] = (values["a"], values["b"])

        # Process dependent variables (non-tuple)
        for param_name, param_config in sampling_space.items():
            if "_original_name" in param_config and "_is_tuple" not in param_config:
                original_name = param_config["_original_name"]
                parent_name = param_config["_parent_name"]
                parent_value = param_config["_parent_value"]
                if suggested_params[parent_name] == parent_value:
                    if param_config["type"] == "categorical":
                        value = trial.suggest_categorical(
                            param_name,
                            param_config["choice"]
                        )
                    elif param_config["type"] == "int":
                        value = trial.suggest_int(
                            param_name,
                            param_config["min"],
                            param_config["max"],
                            step=param_config.get("step", 1),
                            log=param_config.get("log", False)
                        )
                    elif param_config["type"] == "float":
                        value = trial.suggest_float(
                            param_name,
                            param_config["min"],
                            param_config["max"],
                            step=param_config.get("step", None),
                            log=param_config.get("log", False)
                        )
                    suggested_params[original_name] = value
                elif original_name not in suggested_params:
                    suggested_params[original_name] = None
        
        print("Batch parameters:", suggested_params)

        return suggested_params


class StrategyTuning(TuneBase):

    def __init__(self):
        self._market_data = None

    @property
    def market_data(self):
        if self._market_data is None:
            self._market_data = self.getMarketData()
        return self._market_data

    def _objective(self, trial: optuna.Trial, param_space: Dict[str, Any]) -> float:

        # use user-defined parameter search space
        optuna_params = self.suggest_params(trial, param_space)

        # validate parameters
        if not self.validate_params(optuna_params):
            raise optuna.TrialPruned()
        
        # update parameters
        self.__dict__.update(optuna_params)
        
        # run backtest
        result = self.run_backtest(plot=False, print_table=False, send_backend=True, market_data=self.market_data)
        score = result["metric"]["Sharpe"]

        return score
    
    def _optimize(self, 
                  param_space: Dict[str, Any] = {}, 
                  n_trials: int = 100, 
                  timeout: int = 600, 
                  n_jobs: int = -1) -> Tuple[Dict[str, Any], float, str]:
        
        objective_partial = partial(self._objective, param_space=param_space)

        study = optuna.create_study(direction="maximize")
        if n_jobs != 1:
            study.optimize(objective_partial, n_trials=n_trials, timeout=timeout)
        else:
            study.optimize(objective_partial, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

        print("Best parameters:", study.best_params)
        print("Best value:", study.best_value)

        return study

    def autotune(self, 
                 param_space: Dict[str, Any] = {}, 
                 n_trials: int = 100, 
                 timeout: int = 600, 
                 n_jobs: int = -1):

        study = self._optimize(param_space, n_trials, timeout, n_jobs)

        # retrieve all experiment results
        trials_df = study.trials_dataframe()

        # run backtest with local best parameters
        self.__dict__.update(study.best_params)
        best_port_ret = self.run_backtest(plot=True, plot_type="line", print_table=False, market_data=self.market_data)

        return best_port_ret, trials_df
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        # validate parameter combinatons
        if "optimizer" in params:
            if params["optimizer"] == "equal-weight":
                if (params["opt_func"] is not None) or (params["cov_method"] is not None):
                    return False
            if params["optimizer"] == "mean-variance":
                accepted_fields = self._field_accepted_value()
                if (params["opt_func"] not in accepted_fields["opt_func"]) or (params["cov_method"] not in accepted_fields["cov_method"]):
                    return False
        return True

    def _field_accepted_value(self):
        categorical_fields = {
            "optimizer": ["equal-weight", "mean-variance"], 
            "opt_func": ["sharpe", "variance"], 
            "cov_method": ["custom", "shrink", "poet", "structured"], 
        }
        return categorical_fields
    

class FactorTuning(TuneBase, ABC):

    def __init__(self):
        self._inst_close = None

    @property
    def inst_close(self):
        if self._inst_close is None:
            self._inst_close = self.dataloader.load_volume_price(
                "stock_daily_hfq", "close", self.universe, 
                self.start_date_lookback, self.end_date
            )
        return self._inst_close

    def _objective(self, trial: optuna.Trial, param_space: Dict[str, Any], params: Dict[str, Any]) -> float:

        if not param_space:
            raise ValueError("Parameter search space is empty!")

        # copy pre-defined parameters
        raw_params = params.copy()

        # use user-defined parameter search space
        optuna_params = self.suggest_params(trial, param_space)
        raw_params.update(optuna_params)
        
        alpha = self.calc(raw_params)
        
        # calculate evaluation metircs
        score = FactorMetric.Information_Coefficient(alpha, self.inst_close, method="spearman")
        
        return score
    
    def _optimize(self, 
                  param_space: Dict[str, Any] = {}, 
                  params: Dict[str, Any] = {}, 
                  n_trials: int = 100, 
                  timeout: int = 600, 
                  n_jobs: int = -1) -> Tuple[Dict[str, Any], float, str]:
        
        objective_partial = partial(self._objective, param_space=param_space, params=params)
        study = optuna.create_study(direction="maximize")
        if n_jobs != 1:
            study.optimize(objective_partial, n_trials=n_trials, timeout=timeout)
        else:
            study.optimize(objective_partial, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
        
        # obtain best parameters
        print("Best parameters:", study.best_params)
        print("Best value:", study.best_value)
        
        return study

    def autotune(self, 
                 param_space: Dict[str, Any] = {}, 
                 params: Dict[str, Any] = {}, 
                 n_trials: int = 100, 
                 timeout: int = 600, 
                 n_jobs: int = -1):
        
        study = self._optimize(param_space, params, n_trials, timeout, n_jobs)
        
        # retrieve all experiment results
        trials_df = study.trials_dataframe()

        best_params = study.best_params
        best_score = study.best_value
        
        final_params = params.copy()
        final_params.update(best_params)

        # use best parameters to generate final alpha_str
        best_alpha_str = self.create_alpha_str(final_params)

        return trials_df, final_params, best_score, best_alpha_str

    @abstractmethod
    def create_alpha_str(self, params: Dict[str, Any]) -> str:
        pass