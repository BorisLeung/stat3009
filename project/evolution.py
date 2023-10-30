# evolution.py
from dataclasses import dataclass

from cv_logs import save_model, ParameterSearch
from predict import predict
from rv import PositiveInt_rv

import matplotlib.pyplot as plt
import scipy.stats as st
from surprise import AlgoBase
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import RandomizedSearchCV

from typing import Sequence


@dataclass
class EvolutionResult:
    model: AlgoBase
    model_name: str
    error: float
    survival: int
    search: RandomizedSearchCV
    parameters: ParameterSearch


class Evolution:
    def __init__(
        self,
        model: AlgoBase,
        init_param_grid: dict,
        const_params: dict = {},
        int_dist: Sequence[str] = [],
        int_min: int = 5,
        metric: str = "rmse",
        model_suffix: str = "",
    ) -> None:
        self.model = model
        self.init_param_grid = init_param_grid
        self.const_params = const_params
        self.int_dist = int_dist
        self.int_min = int_min
        self.evolution_results = []
        self.metric = metric
        self.model_suffix = model_suffix

    def evolve(
        self,
        train_data: DatasetAutoFolds,
        evolutions: int = 5,
        survivals: float | int = 0.25,
        iterations: int = 1000,
        folds: int = 3,
        **kwargs,
    ) -> None:
        full_train_set = train_data.build_full_trainset()
        if isinstance(survivals, float):
            survival_number = int(iterations * survivals)
        elif isinstance(survivals, int):
            survival_number = survivals
        else:
            raise ValueError("Unknwon type for `survivals`.")
        for evolution in range(evolutions):
            print(f"RUNNING EVOLUTION {len(self.evolution_results)}")
            print("======================================================")
            model_name = f"{self.model.__name__}_evo({len(self.evolution_results)+1}){self.model_suffix}"
            if len(self.evolution_results):
                best_distributions = self.evolution_results[
                    -1
                ].parameters.get_best_distribution(survival_number, plot=False)

                param_grid = self.const_params.copy()
                param_grid |= {
                    param: getattr(st, best_distributions[param][0])(
                        *best_distributions[param][2]
                    )
                    for param in best_distributions.keys()
                    if param not in self.int_dist
                    and param not in self.const_params.keys()
                    and best_distributions[param] is not None
                }

                param_grid |= {
                    param: PositiveInt_rv(
                        getattr(st, best_distributions[param][0])(
                            *best_distributions[param][2]
                        ),
                        self.int_min,
                    )
                    for param in self.int_dist
                }
            else:
                param_grid = self.init_param_grid

            rs = RandomizedSearchCV(
                self.model,
                param_grid,
                measures=[self.metric],
                n_iter=iterations,
                cv=folds,
                n_jobs=-1,
                joblib_verbose=5,
                **kwargs,
            )
            rs.fit(train_data)
            ps = ParameterSearch(rs.cv_results)
            ps.write(f"{model_name}.csv")

            best_model = self.model(**rs.best_params[self.metric])
            best_model.fit(full_train_set)
            predict(best_model, f"{model_name}.csv")
            save_model(best_model, model_name)

            self.evolution_results.append(
                EvolutionResult(
                    best_model,
                    model_name,
                    rs.best_score[self.metric],
                    survival_number,
                    rs,
                    ps,
                )
            )

    def plot_errors(self) -> None:
        errors = []
        for evolution_result in self.evolution_results:
            errors.append(evolution_result.search.best_score[self.metric])
        plt.figure(figsize=(5, 3))
        plt.plot(list(range(1, len(errors) + 1)), errors)
        plt.xlabel("Evolutions")
        plt.ylabel(self.metric)
        plt.show()

    def plot_param(self, parameter: str) -> None:
        column_name = f"param_{parameter}"
        data = []
        for evolution_result in self.evolution_results:
            parameter_df = evolution_result.parameters.results.sort_values(
                by=f"rank_test_{self.metric}"
            )[column_name]
            data.append(parameter_df.head(evolution_result.survival).values)
        figure = plt.figure(figsize=(5, 3))
        ax = figure.add_axes([0, 0, 1, 1])
        ax.boxplot(data)
        plt.show()
