# cv_logs.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
from surprise import AlgoBase

import os
import pickle
from typing import Sequence


class ParameterSearch:
    LOGS_DIR = "cv"

    def __init__(self, cv_results: dict | str) -> None:
        if isinstance(cv_results, dict):
            self.results = pd.DataFrame(cv_results)
        elif isinstance(cv_results, str):
            self.results = pd.read_csv(cv_results)
        else:
            raise ValueError("cv_results with unknown type!")

    def write(self, filename: str, index: bool = False, **kwargs) -> None:
        self.results.to_csv(
            os.path.join(self.LOGS_DIR, filename), index=index, **kwargs
        )

    def plot_params(self, limit: float | int = 0.25) -> None:
        if isinstance(limit, float):
            filtered_results = self.results[
                self.results["rank_test_rmse"] <= (limit * self.results.shape[0])
            ]
        elif isinstance(limit, int):
            filtered_results = self.results[self.results["rank_test_rmse"] <= limit]
        else:
            raise ValueError("percentage with unknown type!")
        PARAMETER_PREFIX = "param_"
        filtered_results = filtered_results[
            [
                column
                for column in filtered_results
                if column.startswith(PARAMETER_PREFIX)
            ]
        ]

        filtered_results = filtered_results.rename(
            columns=lambda column_name: column_name[len(PARAMETER_PREFIX) :]
        )
        for plot_id, column in enumerate(filtered_results):
            plt.subplot(int(filtered_results.shape[0] / 3) + 1, 3, plot_id + 1)
            filtered_results.boxplot([column])

    def get_best_distribution(
        self,
        limit: float | int = 0.25,
        constants: Sequence[str] = [],
        plot: bool = True,
    ) -> dict:
        if isinstance(limit, float):
            filtered_results = self.results[
                self.results["rank_test_rmse"] <= (limit * self.results.shape[0])
            ]
        elif isinstance(limit, int):
            filtered_results = self.results[self.results["rank_test_rmse"] <= limit]
        else:
            raise ValueError("percentage with unknown type!")
        PARAMETER_PREFIX = "param_"
        filtered_results = filtered_results[
            [
                column
                for column in filtered_results
                if column.startswith(PARAMETER_PREFIX)
            ]
        ]

        filtered_results = filtered_results.rename(
            columns=lambda column_name: column_name[len(PARAMETER_PREFIX) :]
        )

        def get_best_distribution(data: np.array):
            if data.dtype not in (int, float):
                return None
            dist_names = [
                "norm",
                "bradford",
                "pareto",
                "alpha",
                "arcsine",
                "dweibull",
                "expon",
                "t",
                "triang",
                "uniform",
                "wrapcauchy",
            ]
            dist_results = []
            params = {}

            for dist_name in dist_names:
                try:
                    dist = getattr(st, dist_name)
                    param = dist.fit(data)

                    params[dist_name] = param
                    # Applying the Kolmogorov-Smirnov test
                    D, p = st.kstest(data, dist_name, args=param)
                    # print("p value for " + dist_name + " = " + str(p))
                    dist_results.append((dist_name, p))
                except:
                    print(f"Error fitting {dist_name}")

            # select the best fitted distribution
            best_dist, best_p = max(dist_results, key=lambda item: item[1])
            # store the name of the best fit and its p value

            # print("Best fitting distribution: " + str(best_dist))
            # print("Best p value: " + str(best_p))
            # print("Parameters for the best fit: " + str(params[best_dist]))

            return best_dist, best_p, params[best_dist]

        best_distributions = {
            column: get_best_distribution(filtered_results[column].values)
            for column in filtered_results
            if column not in constants
        }

        if plot:
            plt.rcParams["figure.figsize"] = [10, 20]
            plt.rcParams["figure.autolayout"] = True
            # plt.figure(figsize=(10, 20))
            # plt.subplots_adjust(wspace=0.6, hspace=0.6)
            plot_id = 1
            for column, distribution in best_distributions.items():
                if distribution is not None:
                    (distribution, p_value, parameters) = distribution
                    ax = plt.subplot(
                        int(len(best_distributions.keys()) / 3) + 1, 3, plot_id
                    )
                    plot_id += 1
                    values = filtered_results[column].values
                    x = np.linspace(values.min(), values.max(), 100)
                    ax.hist(values)
                    ax2 = ax.twinx()
                    ax2.plot(
                        x, getattr(st, distribution)(*parameters).pdf(x), color="red"
                    )
                    ax.set_title(column)

        return best_distributions


MODEL_DIR = "models"


def save_model(model: AlgoBase, model_name: str) -> None:
    with open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), "wb") as file:
        pickle.dump(model, file)


def load_model(model_name: str) -> AlgoBase:
    with open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), "rb") as file:
        obj = pickle.load(file)
    return obj
