import numpy as np
import torch

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner


class SimpleTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):
    """Simple Runner: select forward features and target features. This runner can cover most cases."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C]

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target feature.
        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.
        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """
        # select feature using self.target_features
        # Ensure data has at least 4 dimensions for indexing
        if data.ndim < 4:
            raise ValueError(f"Expected data to have at least 4 dimensions, but got {data.ndim}")

        # Check if target_features is valid for the given data shape
        if self.target_features is not None:
            # If target_features is an integer or slice, directly apply
            if isinstance(self.target_features, (int, slice)):
                # Ensure the last dimension is accessible
                if isinstance(self.target_features, int) and data.shape[-1] <= self.target_features:
                    raise IndexError(
                        f"Target feature index {self.target_features} out of bounds for data with last dimension size {data.shape[-1]}")
                if isinstance(self.target_features, slice):
                    # Basic check for slice bounds
                    stop_idx = self.target_features.stop if self.target_features.stop is not None else float('inf')
                    if stop_idx != float('inf') and data.shape[-1] < stop_idx:
                        raise IndexError(
                            f"Target feature slice stop {self.target_features.stop} out of bounds for data with last dimension size {data.shape[-1]}")
                data = data[:, :, :, self.target_features]
            # !!! 修改这里：在类型检查中加入 torch.Tensor
            elif isinstance(self.target_features, (list, tuple, np.ndarray, torch.Tensor)):
                # Convert to tensor for advanced indexing if not already
                # !!! 这一步仍然执行，但现在 isinstance 检查包含了 torch.Tensor
                if not isinstance(self.target_features, torch.Tensor):
                    self.target_features = torch.tensor(self.target_features, device=data.device, dtype=torch.long)

                # Check if all indices are within bounds
                if not (self.target_features < data.shape[-1]).all():
                    raise IndexError(
                        f"Some target feature indices {self.target_features} are out of bounds for data with last dimension size {data.shape[-1]}")

                data = data[:, :, :, self.target_features]
            else:
                # !!! 这个 else 分支现在应该很难被触发，除非 target_features 是其他未预期的类型
                raise TypeError(
                    f"Unsupported type for target_features: {type(self.target_features)}. Must be int, slice, list, tuple, numpy.ndarray or torch.Tensor.")

        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.
        Args:
            data (tuple): data (future data, history ata).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.
        Returns:
            tuple: (prediction, real_value, (ssl_pred, ssl_true))
            注意：这里返回了 SSL 相关数据
        """
        # preprocess
        future_data, history_data = data
        history_data = self.to_running_device(history_data)  # B, L, N, C
        future_data = self.to_running_device(future_data)  # B, L, N, C

        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)

        # 在训练和非训练模式下都选择输入特征
        future_data_4_dec = self.select_input_features(future_data)
        if not train:
          future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # curriculum learning
        # 从模型获取主任务预测和 SSL 相关数据
        main_prediction_data, ssl_data,cl_loss = self.model(history_data=history_data, future_data=future_data_4_dec,
                                                    batch_seen=iter_num, epoch=epoch, train=train)

        # 检查主任务预测的形状
        assert list(main_prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            "Output shape error, please reshape the forward function output to [B, L, N, C]"

        # 后处理：选择目标特征
        prediction = self.select_target_features(main_prediction_data)
        real_value = self.select_target_features(future_data)

        # 返回主任务的预测、真实值和 SSL 相关数据
        return prediction, real_value, ssl_data, cl_loss


