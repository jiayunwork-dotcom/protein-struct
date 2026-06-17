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

        # ========== 分支2：倾向性特征预测 (更可靠的基础) ==========
        # 使用与enhanced_chou_fasman相同的逻辑：降低中心残基权重，加入beta-over-alpha修正
        prop_probs = np.zeros((n, 3), dtype=np.float64)

        # 预计算每个残基的Pb-Pa差值（用于beta-over-alpha修正）
        beta_pref = np.zeros(n)
        for i in range(n):
            aa = sequence[i]
            if aa in CHOU_FASMAN:
                pa, pb, pc = CHOU_FASMAN[aa]
                beta_pref[i] = pb - pa

        for i in range(n):
            log_scores = self._prior_log.copy()
            prop_feats = self._window_propensity_features(sequence, i)

            # 调整为enhanced_cf相同的权重比例：中心25%，短窗口45%，长窗口30%
            # (原来中心50%导致Gly/Ser的Pc过度主导)
            w_c, w_s, w_l = 0.25, 0.45, 0.30
            log_scores[0] += w_c * prop_feats[6] + w_s * prop_feats[0] + w_l * prop_feats[3]
            log_scores[1] += w_c * prop_feats[7] + w_s * prop_feats[1] + w_l * prop_feats[4]
            log_scores[2] += w_c * prop_feats[8] + w_s * prop_feats[2] + w_l * prop_feats[5]

            # beta-over-alpha偏好修正（与enhanced_cf保持一致）
            # 计算短窗口beta偏好
            start_s = max(0, i - 3)
            end_s = min(n, i + 4)
            short_bp = 0.0
            short_w_sum = 0.0
            for j in range(start_s, end_s):
                dist = abs(j - i)
                w = np.exp(-(dist ** 2) / 8.0)
                short_bp += w * beta_pref[j]
                short_w_sum += w
            if short_w_sum > 0:
                short_bp /= short_w_sum

            # 长窗口beta偏好
            start_l = max(0, i - 5)
            end_l = min(n, i + 6)
            long_bp = 0.0
            long_w_sum = 0.0
            for j in range(start_l, end_l):
                dist = abs(j - i)
                w = np.exp(-(dist ** 2) / 32.0)
                long_bp += w * beta_pref[j]
                long_w_sum += w
            if long_w_sum > 0:
                long_bp /= long_w_sum

            avg_bp = 0.55 * short_bp + 0.45 * long_bp
            if avg_bp > 0.01:
                strength = min(avg_bp * 1.4, 0.75)
                log_scores[1] += strength
                log_scores[0] -= strength * 0.75
            elif avg_bp < -0.01:
                strength = min(-avg_bp * 1.4, 0.75)
                log_scores[0] += strength
                log_scores[1] -= strength * 0.75

            log_scores = log_scores - log_scores.max()
            exp_scores = np.exp(log_scores)
            prop_probs[i] = exp_scores / exp_scores.sum()

        # ========== 分支3：Chou-Fasman 增强预测 ==========
        cf_probs = self._cf_predict(sequence)

        # ========== 融合三个分支 ==========
        # 倾向性特征(对数似然比窗口，基于真实文献数据)是主力
        # CF是辅助，ML分支只做极轻的平滑
        w_nn = 0.05
        w_prop = 0.60
        w_cf = 0.35

        combined = (
            w_nn * nn_probs
            + w_prop * prop_probs
            + w_cf * cf_probs
        )
        combined = combined / combined.sum(axis=1, keepdims=True)

        # ========== 应用生物学约束规则 ==========
        combined = apply_structural_rules(sequence, combined, "nn")

        return PredictionResult(
            sequence=sequence,
            method=self.name,
            probabilities=combined.astype(np.float32),
        )
