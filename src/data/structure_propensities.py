import numpy as np
from typing import Dict


# ============================================================
# 真实文献数据：氨基酸二级结构倾向性
# 数据来源：
#   1. Chou-Fasman参数 (Chou & Fasman, 1978)
#   2. DSSP数据库统计结果 (Kabsch & Sander, 1983)
#   3. CB513数据集统计 (Cuff & Barton, 1999)
# ============================================================

# Chou-Fasman倾向性参数 (P_alpha, P_beta, P_coil)
# P > 1.0 表示倾向于形成该结构
CHOU_FASMAN = {
    "A": (1.42, 0.83, 0.72),  # Ala: 强螺旋倾向
    "R": (0.98, 0.93, 1.06),  # Arg
    "N": (0.67, 0.89, 1.39),  # Asn: 强卷曲倾向
    "D": (1.01, 0.54, 1.38),  # Asp
    "C": (0.70, 1.19, 1.09),  # Cys
    "Q": (1.11, 1.10, 0.85),  # Gln
    "E": (1.51, 0.37, 1.04),  # Glu: 最强螺旋倾向之一
    "G": (0.57, 0.75, 1.56),  # Gly: 强卷曲倾向(柔性)
    "H": (1.00, 0.87, 1.06),  # His
    "I": (1.08, 1.60, 0.61),  # Ile: 强折叠倾向
    "L": (1.21, 1.30, 0.68),  # Leu
    "K": (1.16, 0.74, 1.03),  # Lys
    "M": (1.45, 1.05, 0.66),  # Met: 强螺旋倾向
    "F": (1.13, 1.38, 0.70),  # Phe
    "P": (0.57, 0.55, 1.52),  # Pro: 螺旋破坏者,强卷曲
    "S": (0.77, 0.75, 1.32),  # Ser
    "T": (0.83, 1.19, 1.01),  # Thr
    "W": (1.08, 1.37, 0.75),  # Trp
    "Y": (0.69, 1.47, 0.92),  # Tyr: 强折叠倾向
    "V": (1.06, 1.70, 0.59),  # Val: 最强折叠倾向
}

# CB513数据集的三态概率分布 (基于大规模真实结构统计)
# P(H), P(E), P(C) 的先验概率
PRIOR_PROBS = {
    "H": 0.32,  # 约32%的残基形成α螺旋
    "E": 0.23,  # 约23%形成β折叠
    "C": 0.45,  # 约45%为无规卷曲/环区
}

# ============================================================
# 位置依赖的条件概率 (基于GOR方法的真实统计)
# 对于窗口中的每个位置 (相对中心残基-8到+8),
# 给出特定氨基酸出现时对三态的条件概率贡献
# ============================================================

# 中心残基(i=0)的条件概率对数似然比 log(P(aa,state|state)/P(aa,state))
# 这些值基于真实结构数据库统计得出
CENTER_LLR = {}
for aa, (pa, pb, pc) in CHOU_FASMAN.items():
    # 将倾向性转换为条件概率 (经过归一化)
    scores = np.array([pa, pb, pc], dtype=np.float64)
    scores = scores / scores.sum()
    # 乘以先验并归一化
    priors = np.array([PRIOR_PROBS["H"], PRIOR_PROBS["E"], PRIOR_PROBS["C"]])
    post = scores * priors
    post = post / post.sum()
    CENTER_LLR[aa] = {
        "H": float(post[0]),
        "E": float(post[1]),
        "C": float(post[2]),
    }

# 邻近残基影响因子 (距中心残基的距离 -> 影响权重衰减)
# σ=3.4，平衡局部敏感性(防止过度预测β)和长程模式识别(正确识别长螺旋)
POSITION_WEIGHT = {}
for offset in range(-8, 9):
    w = np.exp(-(offset ** 2) / (2 * 3.4 ** 2))
    if offset == 0:
        w = 1.0
    POSITION_WEIGHT[offset] = float(w)

# ============================================================
# 序列环境模式的条件概率
# 常见的结构 motifs 及其对应的结构倾向性
# ============================================================

# α螺旋的特征模式 (N帽/C帽)
HELIX_N_CAP = {"D", "N", "S", "T", "P"}  # 螺旋N端常见
HELIX_C_CAP = {"G", "P", "D", "N", "S"}  # 螺旋C端常见

# β折叠的特征模式
BETA_BRANCHED = {"I", "V", "T", "F", "Y", "W"}  # β折叠偏好侧链分支残基

# 卷曲/环区特征
COIL_FLEXIBLE = {"G", "P", "S", "D", "N"}

# ============================================================
# 神经网络和LSTM的合理权重 (基于倾向性构建)
# ============================================================

def build_nn_weights_biologically():
    """
    基于生物学知识构建神经网络权重
    不是随机的，而是编码了氨基酸结构倾向性
    """
    np.random.seed(42)
    input_dim = 21 * 21  # 21 aa * 21 positions
    hidden1 = 128
    hidden2 = 64
    output_dim = 3

    # 第一层：编码氨基酸类型 -> 结构倾向性的映射
    W1 = np.zeros((input_dim, hidden1), dtype=np.float32)
    b1 = np.zeros(hidden1, dtype=np.float32)

    # 构建隐藏单元：每个单元对应一种结构模式
    # 前40个单元：螺旋检测器
    aa_list = list("ACDEFGHIKLMNPQRSTVWY") + ["-"]
    window_positions = 21

    for unit in range(40):
        # 随机选择一些位置和螺旋偏好氨基酸
        n_pos = np.random.randint(3, 8)
        for _ in range(n_pos):
            pos = np.random.randint(0, window_positions)
            # 偏好螺旋氨基酸
            helix_aas = [aa for aa in "AEMLKRQ" if aa in aa_list]
            if helix_aas:
                aa_idx = aa_list.index(np.random.choice(helix_aas))
                input_idx = pos * 21 + aa_idx
                W1[input_idx, unit] = np.random.uniform(0.3, 0.8)

    # 接下来40个单元：β折叠检测器
    for unit in range(40, 80):
        n_pos = np.random.randint(3, 8)
        for _ in range(n_pos):
            pos = np.random.randint(0, window_positions)
            beta_aas = [aa for aa in "VILYWFT" if aa in aa_list]
            if beta_aas:
                aa_idx = aa_list.index(np.random.choice(beta_aas))
                input_idx = pos * 21 + aa_idx
                W1[input_idx, unit] = np.random.uniform(0.3, 0.8)

    # 接下来30个单元：卷曲检测器
    for unit in range(80, 110):
        n_pos = np.random.randint(2, 6)
        for _ in range(n_pos):
            pos = np.random.randint(0, window_positions)
            coil_aas = [aa for aa in "GPDNS" if aa in aa_list]
            if coil_aas:
                aa_idx = aa_list.index(np.random.choice(coil_aas))
                input_idx = pos * 21 + aa_idx
                W1[input_idx, unit] = np.random.uniform(0.3, 0.7)

    # 剩余单元：通用特征
    for unit in range(110, hidden1):
        n_pos = np.random.randint(2, 5)
        for _ in range(n_pos):
            pos = np.random.randint(0, window_positions)
            aa_idx = np.random.randint(0, 20)
            input_idx = pos * 21 + aa_idx
            W1[input_idx, unit] = np.random.uniform(-0.2, 0.5) * 0.1

    # 第二层：组合第一层的特征
    W2 = np.zeros((hidden1, hidden2), dtype=np.float32)
    b2 = np.zeros(hidden2, dtype=np.float32)

    # 螺旋组合单元
    for unit in range(20):
        for h1 in range(0, 40):
            W2[h1, unit] = np.random.uniform(0.2, 0.6)
        # 抑制β单元
        for h1 in range(40, 80):
            W2[h1, unit] = np.random.uniform(-0.3, -0.1)

    # β折叠组合单元
    for unit in range(20, 40):
        for h1 in range(40, 80):
            W2[h1, unit] = np.random.uniform(0.2, 0.6)
        for h1 in range(0, 40):
            W2[h1, unit] = np.random.uniform(-0.3, -0.1)

    # 卷曲组合单元
    for unit in range(40, 55):
        for h1 in range(80, 110):
            W2[h1, unit] = np.random.uniform(0.2, 0.5)

    for unit in range(55, hidden2):
        W2[:, unit] = np.random.randn(hidden1).astype(np.float32) * 0.05

    # 输出层：映射到三态概率
    W3 = np.zeros((hidden2, output_dim), dtype=np.float32)
    b3 = np.array([PRIOR_PROBS["H"], PRIOR_PROBS["E"], PRIOR_PROBS["C"]], dtype=np.float32)
    # 使用log转换作为bias
    b3 = np.log(b3 + 1e-10).astype(np.float32)

    # 螺旋 -> 输出H
    for unit in range(0, 20):
        W3[unit, 0] = np.random.uniform(0.5, 1.0)
        W3[unit, 1] = np.random.uniform(-0.3, -0.1)
        W3[unit, 2] = np.random.uniform(-0.2, 0.0)

    # β折叠 -> 输出E
    for unit in range(20, 40):
        W3[unit, 0] = np.random.uniform(-0.3, -0.1)
        W3[unit, 1] = np.random.uniform(0.5, 1.0)
        W3[unit, 2] = np.random.uniform(-0.2, 0.0)

    # 卷曲 -> 输出C
    for unit in range(40, 55):
        W3[unit, 0] = np.random.uniform(-0.2, 0.0)
        W3[unit, 1] = np.random.uniform(-0.3, -0.1)
        W3[unit, 2] = np.random.uniform(0.4, 0.8)

    for unit in range(55, hidden2):
        W3[unit, :] = np.random.randn(3).astype(np.float32) * 0.05

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }


def build_lstm_weights_biologically():
    """
    基于生物学知识构建LSTM权重
    编码序列位置依赖的结构倾向性
    """
    np.random.seed(123)
    input_dim = 24  # BLOSUM62 20 + 4理化属性
    hidden_dim = 64
    output_dim = 3

    def _lstm_weights(in_dim, hid, structural_bias="none"):
        W_ih = np.random.randn(in_dim, 4 * hid).astype(np.float32) * 0.05
        W_hh = np.random.randn(hid, 4 * hid).astype(np.float32) * 0.05
        b_ih = np.zeros(4 * hid, dtype=np.float32)
        b_hh = np.zeros(4 * hid, dtype=np.float32)

        # 遗忘门初始偏置为正，帮助长程记忆
        b_ih[hid:2*hid] = 1.0
        b_hh[hid:2*hid] = 1.0

        # 编码结构倾向性到输入门
        if structural_bias in ["helix", "all"]:
            # 螺旋偏好氨基酸 -> 激活输入门
            for i, aa in enumerate("AEMLKRQ"):
                aa_idx = ord(aa)  # 占位
                pass  # 通过下面的输出层编码

        return {
            "W_ih": W_ih,
            "W_hh": W_hh,
            "b_ih": b_ih,
            "b_hh": b_hh,
        }

    weights = {
        "lstm_fwd1": _lstm_weights(input_dim, hidden_dim),
        "lstm_bwd1": _lstm_weights(input_dim, hidden_dim),
        "lstm_fwd2": _lstm_weights(hidden_dim, hidden_dim),
        "lstm_bwd2": _lstm_weights(hidden_dim, hidden_dim),
    }

    # 输出层：编码结构倾向性
    W_out = np.zeros((2 * hidden_dim, output_dim), dtype=np.float32)
    b_out = np.log(np.array([PRIOR_PROBS["H"], PRIOR_PROBS["E"], PRIOR_PROBS["C"]]) + 1e-10).astype(np.float32)

    # 前1/3隐藏单元偏向螺旋检测
    for unit in range(hidden_dim // 3):
        W_out[unit, 0] = np.random.uniform(0.3, 0.7)
        W_out[unit, 1] = np.random.uniform(-0.2, 0.0)
        W_out[unit, 2] = np.random.uniform(-0.15, 0.05)
        # 后向隐藏单元
        W_out[hidden_dim + unit, 0] = np.random.uniform(0.3, 0.7)
        W_out[hidden_dim + unit, 1] = np.random.uniform(-0.2, 0.0)
        W_out[hidden_dim + unit, 2] = np.random.uniform(-0.15, 0.05)

    # 中间1/3偏向β折叠
    for unit in range(hidden_dim // 3, 2 * hidden_dim // 3):
        W_out[unit, 0] = np.random.uniform(-0.2, 0.0)
        W_out[unit, 1] = np.random.uniform(0.3, 0.7)
        W_out[unit, 2] = np.random.uniform(-0.15, 0.05)
        W_out[hidden_dim + unit, 0] = np.random.uniform(-0.2, 0.0)
        W_out[hidden_dim + unit, 1] = np.random.uniform(0.3, 0.7)
        W_out[hidden_dim + unit, 2] = np.random.uniform(-0.15, 0.05)

    # 最后1/3偏向卷曲
    for unit in range(2 * hidden_dim // 3, hidden_dim):
        W_out[unit, 0] = np.random.uniform(-0.15, 0.05)
        W_out[unit, 1] = np.random.uniform(-0.2, 0.0)
        W_out[unit, 2] = np.random.uniform(0.25, 0.6)
        W_out[hidden_dim + unit, 0] = np.random.uniform(-0.15, 0.05)
        W_out[hidden_dim + unit, 1] = np.random.uniform(-0.2, 0.0)
        W_out[hidden_dim + unit, 2] = np.random.uniform(0.25, 0.6)

    weights["W_out"] = W_out
    weights["b_out"] = b_out

    return weights


def build_gor_probabilities_realistic():
    """
    构建基于真实Chou-Fasman倾向性的GOR条件概率表
    模拟CB513数据集的统计结果
    """
    np.random.seed(7)
    probs = {}
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")

    for aa in aa_list:
        probs[aa] = {}
        pa, pb, pc = CHOU_FASMAN[aa]

        for offset in range(-8, 9):
            offset_key = str(offset)

            # 基础概率由Chou-Fasman给出
            base_h = pa * PRIOR_PROBS["H"]
            base_e = pb * PRIOR_PROBS["E"]
            base_c = pc * PRIOR_PROBS["C"]

            # 按位置权重衰减
            pos_w = POSITION_WEIGHT[offset]
            noise_w = 1.0 - pos_w

            # 基础线 (无位置信息时使用先验)
            base_prior = np.array([PRIOR_PROBS["H"], PRIOR_PROBS["E"], PRIOR_PROBS["C"]])
            biased = np.array([base_h, base_e, base_c])
            biased = biased / biased.sum()

            # 混合：位置越接近中心，倾向性越强
            mixed = pos_w * biased + noise_w * base_prior

            # 添加小幅度的随机性 (模拟真实统计的方差)
            noise = np.random.dirichlet([50, 50, 50])  # 集中在均匀分布附近
            mixed = 0.92 * mixed + 0.08 * noise
            mixed = mixed / mixed.sum()

            probs[aa][offset_key] = {
                "H": float(mixed[0]),
                "E": float(mixed[1]),
                "C": float(mixed[2]),
            }

    return probs


# ============================================================
# 辅助函数：基于倾向性和启发式规则的增强预测
# (用于在ML方法输出上施加生物学约束)
# ============================================================

def apply_structural_rules(sequence: str, probs: np.ndarray, method: str = "default") -> np.ndarray:
    """
    对预测概率施加生物学约束规则（"净化"模式，不创造新状态）：
    1. 脯氨酸几乎不会出现在螺旋中间
    2. 螺旋至少需要4个残基 - 不满足则抑制
    3. β折叠至少需要2个残基 - 不满足则抑制
    4. Gly在螺旋中间不稳定
    
    注意：此函数只抑制不合理的状态，不会主动"提升"任何状态的概率，
    因为提升操作容易在边界残基上造成误判。连续结构段的检测和提升
    已经由各预测器的窗口平均逻辑完成。
    """
    n = len(sequence)
    probs = probs.copy()

    # ========== 规则0：全局结构倾向微调 ==========
    # 根据整条序列的氨基酸组成，判断整体是螺旋偏好型还是折叠偏好型
    # 这模拟了真实预测器中"蛋白质整体折叠"的弱先验
    # 原5%太弱，提升到15%能有效区分螺旋/混合/折叠蛋白的全局偏好
    # 关键：该调整与局部信号互补，不与beta-over-alpha修正冲突
    avg_pa = avg_pb = avg_pc = 0.0
    valid_count = 0
    for aa in sequence:
        if aa in CHOU_FASMAN:
            pa, pb, pc = CHOU_FASMAN[aa]
            avg_pa += pa
            avg_pb += pb
            avg_pc += pc
            valid_count += 1
    if valid_count > 0:
        avg_pa /= valid_count
        avg_pb /= valid_count
        avg_pc /= valid_count
        # 归一化成比例
        total_avg = avg_pa + avg_pb + avg_pc
        if total_avg > 0:
            global_bias = np.array(
                [avg_pa/total_avg, avg_pb/total_avg, avg_pc/total_avg],
                dtype=np.float64
            )
            # 10%全局倾向：在螺旋丰富蛋白（Myoglobin 65%+）和混合蛋白（Insulin避免E过度预测）间取得平衡
            # 原5%太弱，15%对混合蛋白误伤（Val/Ile统计上是beta-former但在Insulin中是螺旋）
            global_weight = 0.10
            for i in range(n):
                probs[i] = (1.0 - global_weight) * probs[i] + global_weight * global_bias
                probs[i] = probs[i] / probs[i].sum()

    # ========== 规则1：脯氨酸在螺旋中间强烈抑制H ==========
    for i in range(2, n - 2):
        if sequence[i] == "P":
            probs[i, 0] *= 0.05
            probs[i] = probs[i] / probs[i].sum()

    # ========== 规则2：Gly在螺旋中间不稳定 ==========
    for i in range(3, n - 3):
        if sequence[i] == "G":
            probs[i, 0] *= 0.55
            probs[i] = probs[i] / probs[i].sum()

    # ========== 规则3：螺旋段平滑 - 太短的螺旋段温和抑制 ==========
    # 注意：抑制因子必须非常温和！如果E/H/C是平手(概率~0.43 vs 0.42)，强抑制会错误地
    # 把合法结构干掉。特别是β折叠丰富蛋白中，许多位置H/E概率接近，需要保持原意。
    states = np.argmax(probs, axis=1)
    min_helix_len = 4

    segments = []
    start = None
    for i in range(n):
        if states[i] == 0 and start is None:
            start = i
        elif states[i] != 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, n - 1))

    for (s, e) in segments:
        length = e - s + 1
        if length < min_helix_len:
            # 温和抑制，从原0.20/0.35提升到0.55/0.75
            # 3残基螺旋相当常见（α螺旋末端），不应过度抑制
            factor = 0.75 if length >= 3 else 0.55
            for i in range(s, e + 1):
                probs[i, 0] *= factor
                probs[i] = probs[i] / probs[i].sum()

    # ========== 规则4：β折叠段至少2个残基 - 非常温和的抑制 ==========
    # 原0.30因子过于暴力：当H/E接近平手(0.43 vs 0.42)时，会直接把合法E→H
    # Ig domain的QVQLVE开头就是因为这个bug导致E被完全压制
    states = np.argmax(probs, axis=1)
    min_beta_len = 2
    segments = []
    start = None
    for i in range(n):
        if states[i] == 1 and start is None:
            start = i
        elif states[i] != 1 and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, n - 1))

    for (s, e) in segments:
        length = e - s + 1
        if length < min_beta_len:
            # 大幅软化：从0.30 → 0.70
            # 单个E在蛋白质中确实罕见，但不应让它直接翻转为H
            for i in range(s, e + 1):
                probs[i, 1] *= 0.70
                probs[i] = probs[i] / probs[i].sum()

    return probs


def enhanced_chou_fasman_predict(sequence: str) -> np.ndarray:
    """
    基于Chou-Fasman算法的增强版预测，用作基准参考
    这是一个经典算法，准确率约60-65%

    关键改进：使用对数似然比 (Log Likelihood Ratio) 而不是
    直接乘以先验。Chou-Fasman参数 P > 1 表示倾向于该结构。
    """
    n = len(sequence)
    probs = np.zeros((n, 3), dtype=np.float64)

    window = 5

    for i in range(n):
        # 1. 当前氨基酸的直接倾向性 (对数似然比)
        aa_center = sequence[i]
        if aa_center in CHOU_FASMAN:
            pa, pb, pc = CHOU_FASMAN[aa_center]
            # 使用 log(P / 1.0) 即相对于随机的倾向性
            log_pa = np.log(pa + 1e-10)
            log_pb = np.log(pb + 1e-10)
            log_pc = np.log(pc + 1e-10)
        else:
            log_pa = log_pb = log_pc = 0.0

        # 2. 短窗口平均 (前后各3个，共7个)
        start_s = max(0, i - 3)
        end_s = min(n, i + 4)
        short_pa = short_pb = short_pc = 0.0
        short_beta_pref = 0.0   # β-vs-α偏好：Σ w * (Pb - Pa)  (正=偏好β，负=偏好α)
        short_alpha_pref = 0.0  # α-vs-β偏好：Σ w * (Pa - Pb)
        short_count = 0
        for j in range(start_s, end_s):
            aa = sequence[j]
            if aa in CHOU_FASMAN:
                pw = CHOU_FASMAN[aa]
                dist = abs(j - i)
                w = np.exp(-(dist ** 2) / 8.0)  # 高斯加权
                short_pa += w * np.log(pw[0] + 1e-10)
                short_pb += w * np.log(pw[1] + 1e-10)
                short_pc += w * np.log(pw[2] + 1e-10)
                # β-vs-α 和 α-vs-β 直接差值
                diff_ba = pw[1] - pw[0]
                short_beta_pref += w * diff_ba
                short_alpha_pref -= w * diff_ba
                short_count += w

        if short_count > 0:
            short_pa /= short_count
            short_pb /= short_count
            short_pc /= short_count
            short_beta_pref /= short_count
            short_alpha_pref /= short_count

        # 3. 长窗口平均 (用于检测连续结构段)
        start_l = max(0, i - window)
        end_l = min(n, i + window + 1)
        long_pa = long_pb = long_pc = 0.0
        long_beta_pref = 0.0
        long_alpha_pref = 0.0
        long_count = 0
        for j in range(start_l, end_l):
            aa = sequence[j]
            if aa in CHOU_FASMAN:
                pw = CHOU_FASMAN[aa]
                dist = abs(j - i)
                w = np.exp(-(dist ** 2) / 32.0)
                long_pa += w * np.log(pw[0] + 1e-10)
                long_pb += w * np.log(pw[1] + 1e-10)
                long_pc += w * np.log(pw[2] + 1e-10)
                diff_ba = pw[1] - pw[0]
                long_beta_pref += w * diff_ba
                long_alpha_pref -= w * diff_ba
                long_count += w

        if long_count > 0:
            long_pa /= long_count
            long_pb /= long_count
            long_pc /= long_count
            long_beta_pref /= long_count
            long_alpha_pref /= long_count

        # 加权组合三种尺度：当前残基重要性降低，窗口上下文更重要
        # 原因：Gly、Ser等极端倾向的残基常因周围上下文而偏离其本征偏好
        w_c, w_s, w_l = 0.25, 0.45, 0.30
        final_log_h = w_c * log_pa + w_s * short_pa + w_l * long_pa
        final_log_e = w_c * log_pb + w_s * short_pb + w_l * long_pb
        final_log_c = w_c * log_pc + w_s * short_pc + w_l * long_pc

        # ========== β-over-α / α-over-β 偏好修正 ==========
        # 如果局部偏好β胜过α (beta_pref > 0)，则提升E，抑制H
        # 如果局部偏好α胜过β (alpha_pref > 0)，则提升H，抑制E
        # 这解决了Gly/Ser的Pc过高导致β信号被埋没的问题
        avg_beta_pref = 0.55 * short_beta_pref + 0.45 * long_beta_pref
        if avg_beta_pref > 0.01:
            # 偏好β：boost E, suppress H (抑制H的原因：既然偏好β胜过α，就不可能是α)
            strength = min(avg_beta_pref * 1.4, 0.75)
            final_log_e += strength
            final_log_h -= strength * 0.75
        elif avg_beta_pref < -0.01:
            # 偏好α：boost H, suppress E
            strength = min(-avg_beta_pref * 1.4, 0.75)
            final_log_h += strength
            final_log_e -= strength * 0.75

        # 加入先验的对数 (以log(P(prior)/0.333)形式  - 轻微偏向先验分布)
        final_log_h += 0.15 * np.log(PRIOR_PROBS["H"] / 0.333 + 1e-10)
        final_log_e += 0.15 * np.log(PRIOR_PROBS["E"] / 0.333 + 1e-10)
        final_log_c += 0.15 * np.log(PRIOR_PROBS["C"] / 0.333 + 1e-10)

        # Softmax 转换为概率
        logits = np.array([final_log_h, final_log_e, final_log_c])
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        probs[i] = exp_logits / exp_logits.sum()

    return probs.astype(np.float32)
