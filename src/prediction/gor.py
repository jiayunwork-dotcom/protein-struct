import os
import json
import numpy as np

from .base import StructurePrediction, PredictionResult, STRUCTURE_STATES
from ..data.amino_acids import AMINO_ACIDS, AMINO_ACID_INDEX
from ..data.structure_propensities import (
    CHOU_FASMAN,
    PRIOR_PROBS,
    POSITION_WEIGHT,
    apply_structural_rules,
    enhanced_chou_fasman_predict,
)


class GOR4Predictor(StructurePrediction):
    """
    GOR-IV 二级结构预测器 (Garnier-Osguthorpe-Robson, version IV)

    实现原理：信息论方法，使用位置相关的对数似然比。
    对每个残基位置 i，取前后各 8 个残基共 17 个位置的窗口，
    每个位置按距离加权，加总各氨基酸的结构倾向性对数似然比。

    基于真实文献数据：
    - 先验概率: CB513数据集统计 (H:32%, E:23%, C:45%)
    - 氨基酸倾向性: Chou-Fasman参数
    - 位置权重: 高斯衰减 (越接近中心影响越大)
    """
    name = "GOR-IV"

    def __init__(self, prob_file: str = None):
        self.window_size = 8
        # GOR-IV 的四种信息论参数 (简化版只使用单残基信息 I(S;R))
        self._prior_log = np.log(
            np.array([PRIOR_PROBS["H"], PRIOR_PROBS["E"], PRIOR_PROBS["C"]], dtype=np.float64)
        )

    def _single_residue_info(self, aa: str) -> np.ndarray:
        """单个残基的对数似然比 log(P(aa|S) / P(aa))"""
        if aa not in CHOU_FASMAN:
            return np.zeros(3, dtype=np.float64)

        pa, pb, pc = CHOU_FASMAN[aa]
        # Chou-Fasman P > 1 表示倾向于该结构
        # 转换为 log-likelihood ratio
        llr = np.log(np.array([pa, pb, pc], dtype=np.float64) + 1e-10)
        return llr

    def predict(self, sequence: str) -> PredictionResult:
        n = len(sequence)
        if n == 0:
            return PredictionResult(sequence="", method=self.name, states="", probabilities=np.zeros((0, 3)))

        # ========== GOR-IV 核心算法：信息论 ==========
        gor_probs = np.zeros((n, 3), dtype=np.float64)

        # 进一步弱化先验：CF已包含先验、beta修正、连续段检测的完整逻辑
        # GOR主要贡献位置上下文的细微信息，不应带来自己的强偏差
        prior_strength = 0.40
        weakened_prior = prior_strength * self._prior_log

        for i in range(n):
            # 初始化为弱化的先验概率
            log_scores = weakened_prior.copy()

            # 对窗口内的每个位置
            for offset in range(-self.window_size, self.window_size + 1):
                pos = i + offset
                if pos < 0 or pos >= n:
                    continue

                aa = sequence[pos]
                if aa not in CHOU_FASMAN:
                    continue

                # 位置权重 (GOR-IV考虑位置相关性)
                pos_w = POSITION_WEIGHT.get(offset, 0.05)

                # 加上该残基的贡献
                # GOR公式: log P(S|R) ∝ log P(S) + sum_j w_j * log(f(S|R_j))
                llr = self._single_residue_info(aa)
                log_scores += pos_w * llr

            # Softmax归一化
            log_scores = log_scores - log_scores.max()
            exp_scores = np.exp(log_scores)
            gor_probs[i] = exp_scores / exp_scores.sum()

        # ========== Chou-Fasman 增强预测 ==========
        cf_probs = enhanced_chou_fasman_predict(sequence)

        # ========== 融合：GOR + CF 加权平均 ==========
        # v6.6: GOR权重从28%→35%，保持与NN/LSTM的区分度
        # GOR特性：基于信息论，对窗口LLR求和直接反映位置特征
        alpha = 0.35  # GOR窗口LLR权重
        beta = 0.65   # Chou-Fasman权重（核心主体）
        combined = alpha * gor_probs + beta * cf_probs
        combined = combined / combined.sum(axis=1, keepdims=True)

        # ========== v6.6: GOR独有的风格 — β-over-α 偏好修正（文献GOR特性）==========
        # GOR-IV文献中对β折叠的接受阈值略低于神经网络
        # 当E与H概率差距小于0.05时，GOR倾向于β
        for i in range(n):
            h_p = combined[i, 0]
            e_p = combined[i, 1]
            c_p = combined[i, 2]
            # 如果E略低于H（差≤0.06），且周围3个残基有≥1个高Pβ残基 → 给E一个加成
            if 0.02 < h_p - e_p <= 0.06:
                s = max(0, i - 2)
                e = min(n, i + 3)
                has_beta = any(sequence[j] in {"V", "I", "F", "Y", "W", "T"} for j in range(s, e))
                if has_beta:
                    combined[i, 1] *= 1.08
                    combined[i, 0] *= 0.96
                    combined[i] = combined[i] / combined[i].sum()
            # 另一个GOR-IV特性：避免极端高H概率（Q65%文献准确性）
            # 如果H>0.92，向C转移1-2%
            if h_p > 0.92:
                transfer = (h_p - 0.92) * 0.40
                combined[i, 0] -= transfer
                combined[i, 2] += transfer * 0.7
                combined[i, 1] += transfer * 0.3

        # ========== 应用生物学约束规则 ==========
        combined = apply_structural_rules(sequence, combined, "gor4")

        return PredictionResult(
            sequence=sequence,
            method=self.name,
            probabilities=combined.astype(np.float32),
        )
