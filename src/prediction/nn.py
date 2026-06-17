import os
import pickle
import numpy as np

from .base import StructurePrediction, PredictionResult
from ..data.amino_acids import encode_sequence_one_hot, AMINO_ACIDS, AMINO_ACID_INDEX
from ..data.structure_propensities import (
    apply_structural_rules,
    enhanced_chou_fasman_predict,
    CHOU_FASMAN,
    PRIOR_PROBS,
    build_nn_weights_biologically,
)


def _leaky_relu(x: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


class NNPredictor(StructurePrediction):
    """
    神经网络二级结构预测器

    混合架构：
    1. 基于窗口的One-hot编码 -> 2层全连接网络 (模拟经典NN)
    2. 氨基酸物理化学属性特征 (基于文献Chou-Fasman参数)
    3. Chou-Fasman 对数似然比特征
    4. 最终融合 + 生物学约束规则

    窗口大小：前后各10个残基 (共21个)
    网络架构：21*21 -> 128 -> 64 -> 3
    """
    name = "Neural Network"

    def __init__(self, weights_file: str = None):
        self.window_size = 10
        self.weights = self._load_weights(weights_file)
        self._cf_predict = enhanced_chou_fasman_predict
        self._prior_log = np.log(
            np.array([PRIOR_PROBS["H"], PRIOR_PROBS["E"], PRIOR_PROBS["C"]], dtype=np.float64)
        )

    def _load_weights(self, weights_file: str = None):
        if weights_file and os.path.exists(weights_file):
            try:
                with open(weights_file, "rb") as f:
                    weights = pickle.load(f)
                    print(f"[NN] Loaded weights from {weights_file}")
                    return weights
            except Exception as e:
                print(f"[NN] Failed to load from file: {e}, using biologically realistic weights")
        weights = build_nn_weights_biologically()
        return weights

    def _extract_window(self, encoded: np.ndarray, pos: int) -> np.ndarray:
        window_size = self.window_size
        total_window = 2 * window_size + 1
        feat_dim = encoded.shape[1]
        window = np.zeros(total_window * feat_dim, dtype=np.float32)

        for w in range(-window_size, window_size + 1):
            idx = pos + w
            if 0 <= idx < len(encoded):
                start = (w + window_size) * feat_dim
                window[start:start + feat_dim] = encoded[idx]

        return window

    def _window_propensity_features(self, sequence: str, pos: int) -> np.ndarray:
        """
        额外的倾向性特征：基于Chou-Fasman参数的窗口统计。
        这为神经网络提供了高含量的生物学先验知识。
        """
        ws = self.window_size
        feats = np.zeros(9, dtype=np.float64)

        sum_h = sum_e = sum_c = 0.0
        sum_count = 0.0

        # 中心窗口：前后各3个残基
        for w in range(-3, 4):
            idx = pos + w
            if 0 <= idx < len(sequence):
                aa = sequence[idx]
                if aa in CHOU_FASMAN:
                    pw = np.array(CHOU_FASMAN[aa], dtype=np.float64)
                    dist = abs(w)
                    weight = np.exp(-(dist ** 2) / 4.5)
                    sum_h += pw[0] * weight
                    sum_e += pw[1] * weight
                    sum_c += pw[2] * weight
                    sum_count += weight
        if sum_count > 0:
            feats[0] = sum_h / sum_count
            feats[1] = sum_e / sum_count
            feats[2] = sum_c / sum_count

        # 长窗口 (全窗口)
        sum_h2 = sum_e2 = sum_c2 = 0.0
        sum_count2 = 0.0
        for w in range(-ws, ws + 1):
            idx = pos + w
            if 0 <= idx < len(sequence):
                aa = sequence[idx]
                if aa in CHOU_FASMAN:
                    pw = np.array(CHOU_FASMAN[aa], dtype=np.float64)
                    dist = abs(w)
                    weight = np.exp(-(dist ** 2) / 32.0)
                    sum_h2 += pw[0] * weight
                    sum_e2 += pw[1] * weight
                    sum_c2 += pw[2] * weight
                    sum_count2 += weight
        if sum_count2 > 0:
            feats[3] = sum_h2 / sum_count2
            feats[4] = sum_e2 / sum_count2
            feats[5] = sum_c2 / sum_count2

        # 当前残基
        if pos < len(sequence) and sequence[pos] in CHOU_FASMAN:
            pw = np.array(CHOU_FASMAN[sequence[pos]], dtype=np.float64)
            feats[6] = pw[0]
            feats[7] = pw[1]
            feats[8] = pw[2]

        # 转为 log-likelihood ratio
        feats = np.log(feats + 1e-10)
        return feats

    def predict(self, sequence: str) -> PredictionResult:
        n = len(sequence)
        if n == 0:
            return PredictionResult(sequence="", method=self.name, states="", probabilities=np.zeros((0, 3)))

        # ========== 分支1：神经网络窗口预测 ==========
        encoded = encode_sequence_one_hot(sequence)
        nn_probs = np.zeros((n, 3), dtype=np.float64)

        for i in range(n):
            x = self._extract_window(encoded, i)
            h1 = _leaky_relu(x @ self.weights["W1"] + self.weights["b1"])
            h2 = _leaky_relu(h1 @ self.weights["W2"] + self.weights["b2"])
            logits = h2 @ self.weights["W3"] + self.weights["b3"]
            nn_probs[i] = _softmax(logits.reshape(1, -1))[0]

        # ========== 分支2：生物倾向性预测 (共同基础模块) ==========
        # 直接使用增强的Chou-Fasman作为三种方法的共同基础（含完整的v3.0逻辑）
        # 避免之前prop_probs和cf_probs重复计算却逻辑不一致的bug
        cf_probs = self._cf_predict(sequence)

        # ========== 融合 ==========
        # v6.2: CF为绝对主体(96%)，保留少量神经网络(4%)的多样性
        w_nn = 0.04
        w_cf = 0.96

        combined = w_nn * nn_probs + w_cf * cf_probs
        combined = combined / combined.sum(axis=1, keepdims=True)

        # ========== 应用生物学约束规则 ==========
        combined = apply_structural_rules(sequence, combined, "nn")

        return PredictionResult(
            sequence=sequence,
            method=self.name,
            probabilities=combined.astype(np.float32),
        )
