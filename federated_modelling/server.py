"""xgboost_comprehensive: A Flower / XGBoost app."""
import pandas as pd
from logging import INFO
from typing import Dict, List, Optional, Union

import xgboost as xgb
from flwr.common import Context, Parameters, Scalar
from flwr.common.config import unflatten_dict
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from task import replace_keys, transform_dataset_to_dmatrix


class CyclicClientManager(SimpleClientManager):
    """Provides a cyclic client selection rule."""

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""

        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        return [self.clients[cid] for cid in available_cids]


def get_evaluate_fn(test_data, params):
    """Return a function for centralised evaluation (logloss)."""

    def evaluate_fn(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=params)
            for para in parameters.tensors:
                para_b = bytearray(para)

            bst.load_model(para_b)
            eval_results = bst.eval_set(
                evals=[(test_data, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
            # Example: "valid-mlogloss:0.54321"
            logloss = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

            with open(f"./centralised_eval.txt", "a", encoding="utf-8") as fp:
                fp.write(f"Round:{server_round},LogLoss:{logloss}\n")

            return logloss, {"logloss": logloss}

    return evaluate_fn


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (logloss) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    logloss_aggregated = (
        sum([metrics["logloss"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"logloss": logloss_aggregated}
    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


# ---------------------------
# Custom Strategy for Saving XGB Model
# ---------------------------
class SaveXgbStrategy(FedXgbBagging):
    """Custom strategy that saves the aggregated global XGB model."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, "FitRes"]],
        failures: list[Union[tuple[ClientProxy, "FitRes"], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        # Aggregate normally
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert Parameters → Booster
            param_bytes = aggregated_parameters.tensors[0]
            bst = xgb.Booster()
            bst.load_model(bytearray(param_bytes))

            # Save final model
            bst.save_model(self.model_path)
            log(INFO, f"✅ Saved global XGB model at round {server_round} to {self.model_path}")

        return aggregated_parameters, aggregated_metrics


# ---------------------------
# Cyclic version inherits both cyclic behavior + save functionality
# ---------------------------
class SaveCyclicStrategy(SaveXgbStrategy, FedXgbCyclic):
    """Cyclic XGB strategy with model saving."""
    pass


# ---------------------------
# Server Function
# ---------------------------
def server_fn(context: Context):
    cfg = replace_keys(unflatten_dict(context.run_config))
    num_rounds = cfg["num_server_rounds"]
    num_local_round = cfg["num_local_round"]
    fraction_fit = cfg["fraction_fit"]
    fraction_evaluate = cfg["fraction_evaluate"]
    train_method = cfg["train_method"]
    params = cfg["params"]
    centralised_eval = cfg["centralised_eval"]
    cv_method = cfg["cv_method"]

    model_path = f"Models/final_global_model_{num_local_round}_{num_rounds}_{train_method}_{cv_method}.json"

    # Prepare test set if centralized evaluation is enabled
    if centralised_eval:
        test_set = pd.read_csv(cfg["test_data_path"])
        test_set.set_format("numpy")
        test_dmatrix = transform_dataset_to_dmatrix(test_set)

    parameters = Parameters(tensor_type="", tensors=[])

    # Choose strategy based on train_method
    if train_method == "bagging":
        strategy = SaveXgbStrategy(
            evaluate_function=(get_evaluate_fn(test_dmatrix, params) if centralised_eval else None),
            model_path=model_path,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate if not centralised_eval else 0.0,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            evaluate_metrics_aggregation_fn=(evaluate_metrics_aggregation if not centralised_eval else None),
            initial_parameters=parameters,
        )
    else:  # cyclic
        strategy = SaveCyclicStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            initial_parameters=parameters,
            model_path=model_path,
        )

    config = ServerConfig(num_rounds=num_rounds)
    client_manager = CyclicClientManager() if train_method == "cyclic" else None

    return ServerAppComponents(
        strategy=strategy, config=config, client_manager=client_manager
    )


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)
