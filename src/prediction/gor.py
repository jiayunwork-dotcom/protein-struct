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

        # 先验强度0.70（弱化原1.0但保留足够的H偏好以检测螺旋）
        # 同时apply_structural_rules的15%全局偏置会根据序列组成动态调整
        prior_strength = 0.70
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
        # 平衡设置：GOR(70%)提供窗口位置上下文，CF(30%)提供生物倾向性+beta-over-alpha修正
        # 加上apply_structural_rules的15%全局偏置，总效果是：
        #   约59.5% GOR窗口, 25.5% CF倾向性, 15%全局序列组成偏置
        # 这个组合能在螺旋丰富（Myoglobin）、混合（Ubiquitin/Insulin）、折叠丰富（Ig）蛋白间取得平衡
        alpha = 0.70  # GOR窗口LLR权重
        beta = 0.30   # Chou-Fasman权重
        combined = alpha * gor_probs + beta * cf_probs
        combined = combined / combined.sum(axis=1, keepdims=True)

        # ========== 应用生物学约束规则 ==========
        combined = apply_structural_rules(sequence, combined, "gor4")

        return PredictionResult(
            sequence=sequence,
            method=self.name,
            probabilities=combined.astype(np.float32),
        )
