import warnings
import xgboost as xgb 
from flwr.client import Client, ClientApp
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status
)
from flwr.common.config import unflatten_dict
from flwr.common.context import Context

from task import load_data, replace_keys
warnings.filterwarnings("ignore", category=UserWarning)


# Define Flower-Xgb Client and client_fn
class XgbClient(Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        train_method,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.train_method = train_method

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        # Cyclic: return the entire model
        bst = (
            bst_input[
                bst_input.num_boosted_rounds()
                - self.num_local_round : bst_input.num_boosted_rounds()
            ]
            if self.train_method == "bagging"
            else bst_input
        )

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        # Example: "valid-mlogloss:0.54321"
        logloss = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=logloss,  # report logloss as the main loss
            num_examples=self.num_val,
            metrics={"logloss": logloss},  # expose in metrics dict too
        )


def client_fn(context: Context):
    fold_id = context.node_config["fold_id"]

    parse_configs = replace_keys(unflatten_dict(context.run_config))
    data_path = parse_configs["data_path"]
    num_local_round = parse_configs["num_local_round"]
    params = parse_configs["params"]
    train_method = parse_configs["train_method"]
    test_fraction = parse_configs["test_fraction"]
    seed = parse_configs["seed"]
    scaled_lr = parse_configs["scaled_lr"]
    if scaled_lr:
        new_lr = parse_configs["params"]["eta"] / parse_configs["num_folds"]
        parse_configs["params"].update({"eta": new_lr})

    return XgbClient(
        *load_data(
            data_path=data_path,
            fold=fold_id,
            test_fraction=test_fraction,
            seed=seed,
        ),
        num_local_round=num_local_round,
        params=params,
        train_method=train_method,
    )

app = ClientApp(client_fn=client_fn)

