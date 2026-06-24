"""
The implementation of USAD (UnSupervised Anomaly Detection) for PyPOTS.
Paper: https://dl.acm.org/doi/10.1145/3394486.3403392
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from pypots.anomaly_detection.base import BaseNNDetector
from pypots.data.dataset import BaseDataset
from pypots.nn.modules.loss import Criterion, MAE, MSE
from pypots.optim.adam import Adam
from pypots.optim.base import Optimizer


class _USADNetwork(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.input_dim = n_steps * n_features

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model // 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, n_steps, n_features]
        batch_size = x.shape[0]
        # تسطيح البيانات
        x_flat = x.view(batch_size, -1)
        # الترميز وفك الترميز
        z = self.encoder(x_flat)
        x_hat_flat = self.decoder(z)
        # إعادة التشكيل إلى الشكل الأصلي
        x_hat = x_hat_flat.view(batch_size, self.n_steps, self.n_features)
        return x_hat


class USAD(BaseNNDetector):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        anomaly_rate: float,
        n_layers: int = 2,
        d_model: int = 64,
        d_ffn: int = 128,
        dropout: float = 0.1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = MAE,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            anomaly_rate=anomaly_rate,
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.training_loss = training_loss

        # بناء الشبكتين (AE1 و AE2)
        self.ae1 = _USADNetwork(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
        )
        self.ae2 = _USADNetwork(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            dropout=self.dropout,
        )

        self.model = nn.ModuleList([self.ae1, self.ae2])
        self._send_model_to_given_device()
        self._print_model_size()

        # إعداد المحسن
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: dict) -> dict:
        return {"X": data["X"]}

    def _assemble_input_for_validating(self, data: dict) -> dict:
        return {"X": data["X"]}

    def _assemble_input_for_testing(self, data: dict) -> dict:
        return {"X": data["X"]}

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        """تدريب النموذج على البيانات المعطاة."""
        if not isinstance(train_set, dict):
            raise TypeError("train_set must be a dictionary")
        if "X" not in train_set:
            raise KeyError("train_set must contain 'X' key")

        # تحضير مجموعة البيانات
        train_dataset = BaseDataset(
            data=train_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        for epoch in range(1, self.epochs + 1):
            self.ae1.train()
            self.ae2.train()

            epoch_loss = 0.0
            for batch in train_loader:
                # ✅ استخراج X من القائمة بشكل صحيح
                if isinstance(batch, (list, tuple)):
                    X = batch[0]
                else:
                    X = batch["X"]
                
                # تحويل إلى float
                X = X.float()
                X = X.to(self.device)

                # Forward pass
                X_hat1 = self.ae1(X)
                X_hat2 = self.ae2(X_hat1)

                # حساب الخسارة
                loss1 = self.training_loss(X, X_hat1)
                loss2 = self.training_loss(X, X_hat2)
                loss = loss1 + loss2

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            if self.verbose:
                print(f"Epoch {epoch:03d} - Training Loss: {avg_loss:.4f}")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        **kwargs,
    ) -> dict:
        """الكشف عن الشذوذ في بيانات الاختبار."""
        # تحضير مجموعة بيانات الاختبار
        test_dataset = BaseDataset(
            data=test_set,
            return_X_ori=False,
            return_X_pred=False,
            return_y=False,
            file_type=file_type,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.ae1.eval()
        self.ae2.eval()

        all_scores = []

        with torch.no_grad():
            for batch in test_loader:
                # ✅ استخراج X من القائمة بشكل صحيح
                if isinstance(batch, (list, tuple)):
                    X = batch[0]
                else:
                    X = batch["X"]
                
                # تحويل إلى float
                X = X.float()
                X = X.to(self.device)

                X_hat1 = self.ae1(X)
                X_hat2 = self.ae2(X_hat1)

                # حساب درجة الشذوذ
                scores = torch.mean((X - X_hat2) ** 2, dim=(1, 2))
                all_scores.append(scores.cpu().numpy())

        scores = np.concatenate(all_scores, axis=0)
        threshold = np.percentile(scores, (1 - self.anomaly_rate) * 100)
        anomalies = (scores > threshold).astype(int)

        return {
            "anomaly_scores": scores,
            "anomaly_detection": anomalies,
        }