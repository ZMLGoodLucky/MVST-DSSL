import math
import functools
from typing import Tuple, Union, Optional, List

import torch
import numpy as np
from easydict import EasyDict
from easytorch.utils.dist import master_only

from .base_runner import BaseRunner
from ..data import SCALER_REGISTRY
from ..utils import load_pkl
from ..metrics import masked_mae, masked_mape, masked_rmse, masked_wape, masked_mse


class BaseTimeSeriesForecastingRunner(BaseRunner):
    """
    Runner for multivariate time series forecasting datasets.
    Features:
        - Evaluate at pre-defined horizons (1~12 as default) and overall.
        - Metrics: MAE, RMSE, MAPE. Allow customization. The best model is the one with the smallest mae at validation.
        - Support setup_graph for the models acting like tensorflow.
        - Loss: MAE (masked_mae) as default. Allow customization.
        - Support curriculum learning.
        - Users only need to implement the `forward` function.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg["DATASET_NAME"]
        self.null_val = cfg.get("NULL_VAL", np.nan)
        self.dataset_type = cfg.get("DATASET_TYPE", " ")
        self.if_rescale = cfg.get("RESCALE", True)

        self.need_setup_graph = cfg["MODEL"].get("SETUP_GRAPH", False)

        self.scaler = load_pkl("{0}/scaler_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"], cfg.get("RESCALE", True)))

        self.loss = cfg["TRAIN"]["LOSS"]

        self.metrics = cfg.get("METRICS", {
            "MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape, "WAPE": masked_wape, "MSE": masked_mse
        })

        self.cl_param = cfg["TRAIN"].get("CL", None)
        if self.cl_param is not None:
            self.warm_up_epochs = cfg["TRAIN"].CL.get("WARM_EPOCHS", 0)
            self.cl_epochs = cfg["TRAIN"].CL.get("CL_EPOCHS")
            self.prediction_length = cfg["TRAIN"].CL.get("PREDICTION_LENGTH")
            self.cl_step_size = cfg["TRAIN"].CL.get("STEP_SIZE", 1)

        self.if_evaluate_on_gpu = cfg.get("EVAL", EasyDict()).get("USE_GPU", True)
        self.evaluation_horizons = [_ - 1 for _ in cfg.get("EVAL", EasyDict()).get("HORIZONS", range(1, 13))]
        assert min(self.evaluation_horizons) >= 0, "The horizon should start counting from 1."

        # !!! 修改：从模型配置中获取 SSL 和 CL 损失权重 !!!
        self.ssl_lambda = cfg["MODEL"].get("SSL_LAMBDA", 0.05)  # SSL 损失权重
        self.cl_lambda = cfg["MODEL"].get("CL_LAMBDA", 0.1)  # CL 损失权重 (新增)
        # !!!

    def setup_graph(self, cfg: dict, train: bool):
        """Setup all parameters and the computation graph.
        Implementation of many works (e.g., DCRNN, GTS) acts like TensorFlow, which creates parameters in the first feedforward process.
        Args:
            cfg (dict): config
            train (bool): training or inferencing
        """
        dataloader = self.build_test_data_loader(cfg=cfg) if not train else self.build_train_data_loader(cfg=cfg)
        data = next(enumerate(dataloader))[1]  # 获取第一个批次
        # !!! 调用 forward 时，忽略返回的 SSL 和 CL 数据 !!!
        _ = self.forward(data=data, epoch=1, iter_num=0, train=train)
        # !!!

    def count_parameters(self):
        """Count the number of parameters in the model."""

        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("Number of parameters: {0}".format(num_parameters))

    def init_training(self, cfg: dict):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (dict): config
        """

        # setup graph
        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=True)
            self.need_setup_graph = False
        # init training
        super().init_training(cfg)
        # count parameters
        self.count_parameters()
        for key, _ in self.metrics.items():
            self.register_epoch_meter("train_"+key, "train", "{:.6f}")

    def init_validation(self, cfg: dict):
        """Initialize validation.

        Including validation meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_validation(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("val_"+key, "val", "{:.6f}")

    def init_test(self, cfg: dict):
        """Initialize test.

        Including test meters, etc.

        Args:
            cfg (dict): config
        """

        if self.need_setup_graph:
            self.setup_graph(cfg=cfg, train=False)
            self.need_setup_graph = False
        super().init_test(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("test_"+key, "test", "{:.6f}")

    def build_train_dataset(self, cfg: dict):
        """Build train dataset

            There are two types of preprocessing methods in BasicTS,
                1. Normalize across the WHOLE dataset.
                2. Normalize on EACH channel (i.e., calculate the mean and std of each channel).

            The reason why there are two different preprocessing methods is that each channel of the dataset may have a different value range.
                1. Normalizing the WHOLE data set will preserve the relative size relationship between channels.
                   Larger channels usually produce larger loss values, so more attention will be paid to these channels when optimizing the model.
                   Therefore, this approach will achieve better performance when we evaluate on the rescaled dataset.
                   For example, when evaluating rescaled data for two channels with values in the range [0, 1], [9000, 10000], the prediction on channel [0,1] is trivial.
                2. Normalizing each channel will eliminate the gap in value range between channels.
                   For example, a channel with a value in the range [0, 1] may be as important as a channel with a value in the range [9000, 10000].
                   In this case we need to normalize each channel and evaluate without rescaling.

            There is no absolute good or bad distinction between the above two situations,
                  and the decision needs to be made based on actual requirements or academic research habits.
            For example, the first approach is often adopted in the field of Spatial-Temporal Forecasting (STF).
            The second approach is often adopted in the field of Long-term Time Series Forecasting (LTSF).

            To avoid confusion for users and facilitate them to obtain results comparable to existing studies, we
            automatically select data based on the cfg.get("RESCALE") flag (default to True).
            if_rescale == True: use the data that is normalized across the WHOLE dataset
            if_rescale == False: use the data that is normalized on EACH channel

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        data_file_path = "{0}/data_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TRAIN"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", True))
        index_file_path = "{0}/index_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TRAIN"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", True))

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "train"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("train len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        """Build val dataset

        Args:
            cfg (dict): config

        Returns:
            validation dataset (Dataset)
        """
        # see build_train_dataset for details
        data_file_path = "{0}/data_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["VAL"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", True))
        index_file_path = "{0}/index_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["VAL"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", True))

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "valid"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("val len: {0}".format(len(dataset)))

        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        """Build val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        data_file_path = "{0}/data_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TEST"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", True))
        index_file_path = "{0}/index_in_{1}_out_{2}_rescale_{3}.pkl".format(
            cfg["TEST"]["DATA"]["DIR"],
            cfg["DATASET_INPUT_LEN"],
            cfg["DATASET_OUTPUT_LEN"],
            cfg.get("RESCALE", True))

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "test"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("test len: {0}".format(len(dataset)))

        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:
        """Calculate task level in curriculum learning.

        Args:
            epoch (int, optional): current epoch if in training process, else None. Defaults to None.

        Returns:
            int: task level
        """

        if epoch is None:
            return self.prediction_length
        epoch -= 1
        # generate curriculum length
        if epoch < self.warm_up_epochs:
            # still warm up
            cl_length = self.prediction_length
        else:
            _ = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(_, self.prediction_length)
        return cl_length

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test.
        Returns:
            tuple: (prediction, real_value, (ssl_pred, ssl_true), cl_loss).
            *** 注意：子类需要实现此方法，并返回四个值 ***
        """
        raise NotImplementedError()

    def metric_forward(self, metric_func, args):
        """Computing metrics.

        Args:
            metric_func (function, functools.partial): metric function.
            args (list): arguments for metrics computation.
        """

        if isinstance(metric_func, functools.partial) and list(metric_func.keywords.keys()) == ["null_val"]:
            # support partial(metric_func, null_val = something)
            metric_item = metric_func(*args)
        elif callable(metric_func):
            # is a function
            metric_item = metric_func(*args, null_val=self.null_val)
        else:
            raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
        return metric_item

    def rescale_data(self, input_data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Rescale data.

        Args:
            data (List[torch.Tensor]): list of data to be re-scaled.

        Returns:
            List[torch.Tensor]: list of re-scaled data.
        """

        # prediction, real_value = input_data[:2]
        if self.if_rescale:
            input_data[0] = SCALER_REGISTRY.get(self.scaler["func"])(input_data[0], **self.scaler["args"])
            input_data[1] = SCALER_REGISTRY.get(self.scaler["func"])(input_data[1], **self.scaler["args"])
        return input_data

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training details.
        *** 已修改以包含 SSL 和 CL 损失计算 ***
        """
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        # !!! 修改：从 forward 方法获取四个值 !!!
        main_task_prediction, main_task_real_value, ssl_data, cl_loss = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        ssl_pred, ssl_true = ssl_data # 解包 SSL 数据

        # 对主任务的预测和真实值进行重缩放
        main_task_data_for_loss = self.rescale_data([main_task_prediction, main_task_real_value])
        prediction_scaled, real_value_scaled = main_task_data_for_loss[0], main_task_data_for_loss[1]

        # curriculum learning
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            prediction_scaled = prediction_scaled[:, :cl_length, :, :]
            real_value_scaled = real_value_scaled[:, :cl_length, :, :]

        # 计算主任务损失
        total_loss = self.metric_forward(self.loss, [prediction_scaled, real_value_scaled])

        # !!! 如果 ssl_lambda 大于 0 并且 SSL 预测和真实值存在，则计算并添加 SSL 损失
        if ssl_pred is not None and ssl_true is not None and self.ssl_lambda > 0:
            # 这里使用 masked_mae 作为重建损失，您可以根据需要改为 MSELoss 等。
            ssl_loss_val = self.metric_forward(masked_mae, [ssl_pred, ssl_true])
            total_loss += self.ssl_lambda * ssl_loss_val # 将 SSL 损失加到总损失中
            # self.update_epoch_meter("train_ssl_loss", ssl_loss_val.item()) # (可选) 记录 SSL 损失

        # !!! 如果 cl_loss 存在且 cl_lambda 大于 0，则添加对比损失 (新增)
        if cl_loss is not None and self.cl_lambda > 0:
            total_loss += self.cl_lambda * cl_loss # CL 损失通常不需要重缩放
            # self.update_epoch_meter("train_cl_loss", cl_loss.item()) # (可选) 记录 CL 损失

        # 主任务的指标更新
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction_scaled, real_value_scaled])
            self.update_epoch_meter("train_" + metric_name, metric_item.item())

        return total_loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation details.
        *** 已修改以忽略 SSL 和 CL 损失 ***
        """
        # !!! 修改：解包四个值，但只使用前两个 !!!
        prediction, real_value, _, _ = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)

        # 重缩放数据
        forward_return_scaled = self.rescale_data([prediction, real_value])
        prediction_scaled, real_value_scaled = forward_return_scaled[0], forward_return_scaled[1]

        # 指标计算
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction_scaled, real_value_scaled])
            self.update_epoch_meter("val_" + metric_name, metric_item.item())

    def evaluate(self, prediction, real_value):
        """Evaluate the model on test data.

        Args:
            prediction (torch.Tensor): prediction data [B, L, N, C].
            real_value (torch.Tensor): ground truth [B, L, N, C].
        """

        # test performance of different horizon
        for i in self.evaluation_horizons:
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred = prediction[:, i, :, :]
            real = real_value[:, i, :, :]
            # metrics
            metric_repr = ""
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, [pred, real])
                metric_repr += ", Test {0}: {1:.6f}".format(metric_name, metric_item.item())
            log = "Evaluate best model on test data for horizon {:d}" + metric_repr
            log = log.format(i+1)
            self.logger.info(log)
        # test performance overall
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction, real_value])
            self.update_epoch_meter("test_"+metric_name, metric_item.item())

    @torch.no_grad()
    @master_only
    def test(self):
        """Evaluate the model.
        *** 已修改以忽略 SSL 和 CL 损失 ***
        """
        # test loop
        prediction = []
        real_value = []
        for _, data in enumerate(self.test_data_loader):
            # !!! 修改：解包四个值，但只使用前两个 !!!
            preds, testy, _, _ = self.forward(data, epoch=None, iter_num=None, train=False)

            if not self.if_evaluate_on_gpu:
                preds, testy = preds.detach().cpu(), testy.detach().cpu()
            prediction.append(preds)
            real_value.append(testy)

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)

        # 重缩放数据
        if self.if_rescale:
            prediction = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])
            real_value = SCALER_REGISTRY.get(self.scaler["func"])(real_value, **self.scaler["args"])

        # 评估
        self.evaluate(prediction, real_value)


    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        if train_epoch is not None:
            self.save_best_model(train_epoch, "val_MAE", greater_best=False)

