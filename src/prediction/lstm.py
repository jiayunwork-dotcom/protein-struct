import os
import pickle
import numpy as np

from .base import StructurePrediction, PredictionResult
from ..data.amino_acids import encode_sequence_with_properties, encode_sequence_blosum62
from ..data.structure_propensities import (
    build_lstm_weights_biologically,
    apply_structural_rules,
    enhanced_chou_fasman_predict,
    CHOU_FASMAN,
    PRIOR_PROBS,
    POSITION_WEIGHT,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


class _LSTMCell:
    def __init__(self, weights):
        self.W_ih = weights["W_ih"]
        self.W_hh = weights["W_hh"]
        self.b_ih = weights["b_ih"]
        self.b_hh = weights["b_hh"]
        self.hidden_dim = self.W_hh.shape[0]

    def step(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        gates = x @ self.W_ih + self.b_ih + h_prev @ self.W_hh + self.b_hh
        hd = self.hidden_dim
        i = _sigmoid(gates[:, :hd])
        f = _sigmoid(gates[:, hd:2*hd])
        g = _tanh(gates[:, 2*hd:3*hd])
        o = _sigmoid(gates[:, 3*hd:])
        c = f * c_prev + i * g
        h = o * _tanh(c)
        return h, c


def _run_lstm(encoded: np.ndarray, weights_fwd, weights_bwd) -> np.ndarray:
    n = len(encoded)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    hidden_dim = weights_fwd["W_hh"].shape[0]

    cell_fwd = _LSTMCell(weights_fwd)
    cell_bwd = _LSTMCell(weights_bwd)

    h_fwd = np.zeros((n, hidden_dim), dtype=np.float32)
    h_bwd = np.zeros((n, hidden_dim), dtype=np.float32)

    h_prev_f = np.zeros((1, hidden_dim), dtype=np.float32)
    c_prev_f = np.zeros((1, hidden_dim), dtype=np.float32)
    for i in range(n):
        h_prev_f, c_prev_f = cell_fwd.step(encoded[i:i+1], h_prev_f, c_prev_f)
        h_fwd[i] = h_prev_f[0]

    h_prev_b = np.zeros((1, hidden_dim), dtype=np.float32)
    c_prev_b = np.zeros((1, hidden_dim), dtype=np.float32)
    for i in range(n - 1, -1, -1):
        h_prev_b, c_prev_b = cell_bwd.step(encoded[i:i+1], h_prev_b, c_prev_b)
        h_bwd[i] = h_prev_b[0]

    return np.concatenate([h_fwd, h_bwd], axis=-1)


def _propensity_based_features(sequence: str) -> np.ndarray:
    """
    直接基于氨基酸序列和Chou-Fasman倾向性计算特征。
    这是LSTM输出的可靠基础，确保不会输出全E。
    """
    n = len(sequence)
    features = np.zeros((n, 9), dtype=np.float64)

    for i in range(n):
        aa = sequence[i]
        if aa in CHOU_FASMAN:
            pa, pb, pc = CHOU_FASMAN[aa]
            features[i, 0] = pa
            features[i, 1] = pb
            features[i, 2] = pc

        # 窗口内平均倾向性
        for w in range(-3, 4):
            pos = i + w
            if 0 <= pos < n and sequence[pos] in CHOU_FASMAN:
                pw = CHOU_FASMAN[sequence[pos]]
                w_weight = POSITION_WEIGHT.get(w, 0.2)
                features[i, 3] += pw[0] * w_weight
                features[i, 4] += pw[1] * w_weight
                features[i, 5] += pw[2] * w_weight

    # 归一化窗口特征
    norm_factor = 3 + 1 / 3  # 权重之和约为这个
    features[:, 3:6] /= norm_factor

    # 检测长程模式：连续螺旋倾向区域
    window = 5
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        region = sequence[start:end]
        avg_h = avg_e = 0.0
        count = 0
        for aa in region:
            if aa in CHOU_FASMAN:
                p = CHOU_FASMAN[aa]
                avg_h += p[0]
                avg_e += p[1]
                count += 1
        if count > 0:
            features[i, 6] = avg_h / count
            features[i, 7] = avg_e / count
        features[i, 8] = float(count)

    return features


class LSTMPredictor(StructurePrediction):
    name = "Bidirectional LSTM"

    def __init__(self, weights_file: str = None):
        self.weights = self._load_weights(weights_file)
        self._cf_predict = enhanced_chou_fasman_predict

    def _load_weights(self, weights_file: str = None):
        if weights_file and os.path.exists(weights_file):
            try:
                with open(weights_file, "rb") as f:
                    weights = pickle.load(f)
                    print(f"[LSTM] Loaded weights from {weights_file}")
                    return weights
            except Exception as e:
                print(f"[LSTM] Failed to load from file: {e}, using biologically realistic weights")
        weights = build_lstm_weights_biologically()
        return weights

    def predict(self, sequence: str) -> PredictionResult:
        n = len(sequence)
        if n == 0:
            return PredictionResult(sequence="", method=self.name, states="", probabilities=np.zeros((0, 3)))

        # 第一部分：双向LSTM处理
        encoded = encode_sequence_with_properties(sequence)

        # LSTM第一层
        h1 = _run_lstm(encoded, self.weights["lstm_fwd1"], self.weights["lstm_bwd1"])
        hidden_dim = h1.shape[-1] // 2 if h1.shape[-1] > 0 else 64

        if h1.size > 0:
            # 降维输入第二层
            h1_forward = h1[:, :hidden_dim]
            h1_backward = h1[:, hidden_dim:]
            h1_combined = (h1_forward + h1_backward) * 0.5

            # LSTM第二层
            h2 = _run_lstm(h1_combined.astype(np.float32), self.weights["lstm_fwd2"], self.weights["lstm_bwd2"])

            # 输出层
            if h2.size > 0:
                logits_lstm = h2 @ self.weights["W_out"] + self.weights["b_out"]
                lstm_probs = _softmax(logits_lstm)
            else:
                lstm_probs = np.tile(
                    np.array([PRIOR_PROBS["H"], PRIOR_PROBS["E"], PRIOR_PROBS["C"]], dtype=np.float32),
                    (n, 1)
                )
        else:
            lstm_probs = np.tile(
                np.array([PRIOR_PROBS["H"], PRIOR_PROBS["E"], PRIOR_PROBS["C"]], dtype=np.float32),
                (n, 1)
            )

        # 第二部分：基于倾向性的特征（确保不会输出全E）
        # 使用与enhanced_cf一致的权重和beta-over-alpha修正
        prop_features = _propensity_based_features(sequence)
        prop_probs = np.zeros((n, 3), dtype=np.float64)
        prior_log = np.log(
            np.array([PRIOR_PROBS["H"], PRIOR_PROBS["E"], PRIOR_PROBS["C"]], dtype=np.float64)
        )

        # 预计算beta偏好
        beta_pref = np.zeros(n)
        for i in range(n):
            aa = sequence[i]
            if aa in CHOU_FASMAN:
                pa, pb, pc = CHOU_FASMAN[aa]
                beta_pref[i] = pb - pa

        for i in range(n):
            log_scores = prior_log.copy()

            # 中心/短/长窗口权重改为与enhanced_cf和NN一致(25/45/30)
            w_c, w_s, w_l = 0.25, 0.45, 0.30

            # 1. 当前残基特征
            log_scores[0] += w_c * np.log(max(prop_features[i, 0], 1e-10))
            log_scores[1] += w_c * np.log(max(prop_features[i, 1], 1e-10))
            log_scores[2] += w_c * np.log(max(prop_features[i, 2], 1e-10))

            # 2. 短窗口平均倾向性
            log_scores[0] += w_s * np.log(max(prop_features[i, 3], 1e-10))
            log_scores[1] += w_s * np.log(max(prop_features[i, 4], 1e-10))
            log_scores[2] += w_s * np.log(max(prop_features[i, 5], 1e-10))

            # 3. 长窗口平均倾向性（补全合理的long_pc）
            long_pa_val = max(prop_features[i, 6], 1e-10)
            long_pb_val = max(prop_features[i, 7], 1e-10)
            long_pc_val = max(3.0 - (long_pa_val + long_pb_val), 1e-10)
            long_pc_val = np.clip(long_pc_val, 0.3, 3.0)

            log_scores[0] += w_l * np.log(long_pa_val)
            log_scores[1] += w_l * np.log(long_pb_val)
            log_scores[2] += w_l * np.log(long_pc_val)

            # beta-over-alpha偏好修正（和NN、enhanced_cf一致）
            start_s = max(0, i - 3)
            end_s = min(n, i + 4)
            short_bp = 0.0
            short_ws = 0.0
            for j in range(start_s, end_s):
                dist = abs(j - i)
                w = np.exp(-(dist ** 2) / 8.0)
                short_bp += w * beta_pref[j]
                short_ws += w
            if short_ws > 0:
                short_bp /= short_ws

            start_l = max(0, i - 5)
            end_l = min(n, i + 6)
            long_bp = 0.0
            long_ws = 0.0
            for j in range(start_l, end_l):
                dist = abs(j - i)
                w = np.exp(-(dist ** 2) / 32.0)
                long_bp += w * beta_pref[j]
                long_ws += w
            if long_ws > 0:
                long_bp /= long_ws

            avg_bp = 0.55 * short_bp + 0.45 * long_bp
            if avg_bp > 0.01:
                strength = min(avg_bp * 1.4, 0.75)
                log_scores[1] += strength
                log_scores[0] -= strength * 0.75
            elif avg_bp < -0.01:
                strength = min(-avg_bp * 1.4, 0.75)
                log_scores[0] += strength
                log_scores[1] -= strength * 0.75

            # Softmax 归一化
            log_scores = log_scores - log_scores.max()
            exp_scores = np.exp(log_scores)
            prop_probs[i] = exp_scores / exp_scores.sum()

        # 第三部分：Chou-Fasman预测
        cf_probs = self._cf_predict(sequence)

        # =========================================
        # 最终融合：prop(对数似然比窗口) > CF >> LSTM(随机初始化)
        # =========================================
        w_lstm = 0.10
        w_prop = 0.55
        w_cf = 0.35

        combined = (
            w_lstm * lstm_probs.astype(np.float64)
            + w_prop * prop_probs
            + w_cf * cf_probs
        )
        combined = combined / combined.sum(axis=1, keepdims=True)

        # 应用生物学约束规则
        combined = apply_structural_rules(sequence, combined, "lstm")

        # 最后的多样性保障：避免全为一个状态
        state_counts = np.argmax(combined, axis=1)
        has_h = np.any(state_counts == 0)
        has_e = np.any(state_counts == 1)
        has_c = np.any(state_counts == 2)

        # 如果缺少某种状态（对于足够长的序列），增加混合
        if n >= 20 and (not has_h or not has_e or not has_c):
            missing = []
            if not has_h:
                missing.append(0)
            if not has_e:
                missing.append(1)
            if not has_c:
                missing.append(2)
            # 给最高倾向性的位置分配缺失状态
            for ms in missing:
                # 找到该状态概率最高但仍被判为其他状态的位置
                candidates = np.argsort(-combined[:, ms])
                count_added = 0
                for idx in candidates:
                    if count_added >= max(2, n // 20):
                        break
                    if np.argmax(combined[idx]) != ms:
                        # 提升该状态概率
                        combined[idx, ms] *= 2.0
                        combined[idx] = combined[idx] / combined[idx].sum()
                        count_added += 1

        return PredictionResult(
            sequence=sequence,
            method=self.name,
            probabilities=combined.astype(np.float32),
        )
