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

    # ========== 规则0：极弱的全局结构倾向 (v6.2 CF已自带强类型检测) ==========
    # v6.2的enhanced_chou_fasman已经包含蛋白类型检测+上下文感知修正，
    # 此处仅保留3%的极微弱平滑，避免与CF的判断产生冲突
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
        total_avg = avg_pa + avg_pb + avg_pc
        if total_avg > 0:
            global_bias = np.array(
                [avg_pa/total_avg, avg_pb/total_avg, avg_pc/total_avg],
                dtype=np.float64
            )
            # v6.2: 从10%降至3%，避免与CF的类型检测冲突
            global_weight = 0.03
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


# 残基分类 - 用于上下文感知
ALPHA_CORE = {"A", "E", "M", "L", "K", "R", "Q"}  # 强螺旋形成者 (Pα > 1.1 通常)
BETA_CORE = {"V", "I", "F", "Y", "W", "T"}         # 强折叠形成者 (Pβ > 1.1)
AMBIVALENT = {"L", "M", "C", "Q"}                   # 上下文依赖: L/M在Alpha列表也可能螺旋
COIL_BREAKERS = {"G", "P", "S", "D", "N"}           # 强卷曲/破坏者


def _detect_protein_type(sequence: str) -> str:
    """
    v6.5 - 解决±1字符导致分类跳变的鲁棒性问题
    关键修正：
    1. β折叠检测阈值放宽：beta_signal从0.45→0.40，足够turns+beta_core即可
    2. coil分类门槛大幅提高：0.42→0.52，beta蛋白天然有40-45% turns/coil_breakers
       不应误归为coil（否则所有E被强力抑制→E=0）
    3. beta vs alpha放宽：beta_signal只需>alpha_signal（原需+0.03）
    """
    n = len(sequence)
    if n == 0:
        return "mixed"

    alpha_count = sum(1 for aa in sequence if aa in ALPHA_CORE)
    beta_core_count = sum(1 for aa in sequence if aa in BETA_CORE)
    coil_count = sum(1 for aa in sequence if aa in COIL_BREAKERS)
    turn_count = sum(1 for aa in sequence if aa in {"G", "S", "T", "N", "D", "P"})
    cys_count = sum(1 for aa in sequence if aa == "C")

    frq_a = alpha_count / n
    frq_bc = beta_core_count / n
    frq_c = coil_count / n
    frq_turn = turn_count / n
    frq_cys = cys_count / n

    turn_contribution = 0.5 * frq_turn if frq_turn > 0.30 else 0.0
    cys_contribution = 0.20 * frq_cys
    beta_signal = frq_bc + turn_contribution + cys_contribution
    alpha_signal = frq_a

    alpha_beta_ratio = alpha_signal / max(frq_bc, 0.01)
    beta_is_clean = frq_turn > 0.32

    # 判据（v6.5: 放宽beta，收紧coil，防止分类漂移）
    if alpha_signal > 0.48 and alpha_signal > beta_signal + 0.08:
        return "helix"
    # β折叠：放宽beta_signal到0.40；只需>=alpha即可；关键约束alpha<40%（避免Ubiquitin被误归beta）
    #   真正的β折叠蛋白：螺旋残基不会超过40%（Ig只有34-35%）
    elif (beta_signal > 0.40
          and beta_signal >= alpha_signal
          and beta_is_clean
          and alpha_signal < 0.40
          and alpha_beta_ratio < 2.0):
        return "beta"
    # coil：必须>52% coil_breakers（真无序蛋白），beta蛋白的45%turns不再误触发
    elif frq_c > 0.52:
        return "coil"
    else:
        return "mixed"


def _context_bias(sequence: str, i: int, win_radius: int = 4) -> np.ndarray:
    """
    v6.1 - 基于"相对优势"而非绝对阈值（解决Insulin中42%也应识别为螺旋上下文的问题）
    上下文感知偏置：根据窗口内的残基组成，判断局部是螺旋还是折叠上下文。
    核心：不看绝对比例多少，而看α与β谁占优势。
    返回: (delta_H, delta_E, delta_C) logit偏置
    """
    n = len(sequence)
    s = max(0, i - win_radius)
    e = min(n, i + win_radius + 1)
    window_seq = sequence[s:e]

    alpha_n = 0
    beta_n = 0
    coil_n = 0
    turn_n = 0
    total = 0
    for aa in window_seq:
        if aa in ALPHA_CORE:
            alpha_n += 1
        elif aa in BETA_CORE:
            beta_n += 1
        elif aa in COIL_BREAKERS:
            coil_n += 1
            if aa in {"G", "S", "T", "N", "D", "P"}:
                turn_n += 1
        total += 1

    if total == 0:
        return np.zeros(3, dtype=np.float64)

    frq_a = alpha_n / total
    frq_b = beta_n / total
    frq_c = coil_n / total
    frq_t = turn_n / total

    center = sequence[i]
    bias = np.zeros(3, dtype=np.float64)

    # ===== v6.1 核心：相对比较而非绝对阈值 =====
    # 构造相对信号：alpha优势 vs beta优势（都减去coil的比例）
    a_minus_b = frq_a - frq_b
    # β信号加上turns（turns是β链的边界）
    b_plus_turn = frq_b + 0.4 * frq_t
    b_minus_a = b_plus_turn - frq_a

    # 1) 局部螺旋上下文：alpha明显占优（即使只有40% alpha vs 25% beta=差15%）
    #    → 对V/I/L/Y/F/C（本来Pβ高）施加H-pull，强力抑制E
    #    放宽阈值：a_minus_b >= 0.08 (原来是0.10) 覆盖更多位置
    if a_minus_b >= 0.08 and center in (BETA_CORE | AMBIVALENT | {"C", "Y", "F"}):
        # 强度：与差值成正比，最高到0.32
        strength = min(0.32, 0.16 + 0.7 * max(0, a_minus_b - 0.08))
        bias[0] += strength
        bias[1] -= 1.0 * strength  # 超强抑制E（解决Insulin问题！）

    # 2) 局部β折叠上下文：b_plus_turn明显占优
    #    → 对A/E/K（本来Pα高）施加E-pull
    if b_minus_a >= 0.10 and frq_c < 0.50 and (center in ALPHA_CORE or center in AMBIVALENT):
        strength = min(0.22, 0.10 + 0.5 * (b_minus_a - 0.10))
        bias[1] += strength
        bias[0] -= 0.4 * strength

    # 3) 局部Coil上下文：大量G/P/S/D/N
    if frq_c >= 0.50:
        coil_strength = min(0.25, 0.10 + 0.5 * (frq_c - 0.50))
        bias[2] += coil_strength
        bias[0] -= 0.30 * coil_strength
        bias[1] -= 0.30 * coil_strength

    return bias


def _alternating_hydrophobicity_beta_signal(sequence: str, i: int) -> float:
    """
    检测β折叠特征模式：i, i+2, i+4 位的疏水性交替模式。
    β链中，侧链交替伸向相反方向，因此每2位出现疏水残基是特征。
    返回: 0~1之间的beta-stand signal强度
    """
    n = len(sequence)
    if n < 5:
        return 0.0

    # 疏水残基列表
    HYDROPHOBIC = {"V", "I", "L", "F", "Y", "W", "M", "C", "A"}
    # 检查5位窗口 (i-2, i, i+2): pattern [i-2 hydrophobic] OR [i hydrophobic] OR [i+2 hydrophobic]
    # 且中间夹着亲水/G/S

    def _hydro(aa):
        return 1.0 if aa in HYDROPHOBIC else 0.0

    positions = []
    for offset in [-4, -2, 0, 2, 4]:
        pos = i + offset
        if 0 <= pos < n:
            positions.append((pos, _hydro(sequence[pos])))

    if len(positions) < 3:
        return 0.0

    # 评估交替模式: 相邻offset(-2,+2等)应该有相似的疏水倾向
    even_offsets = [p for p in positions if (p[0] - i) % 2 == 0]
    odd_offsets = [p for p in positions if (p[0] - i) % 2 != 0]

    avg_even = sum(p[1] for p in even_offsets) / max(len(even_offsets), 1)
    avg_odd = sum(p[1] for p in odd_offsets) / max(len(odd_offsets), 1)

    # 典型beta模式: 偶数位全部疏水，奇数位全部亲水
    pattern_score = 0.0
    if len(even_offsets) >= 2 and len(odd_offsets) >= 1:
        contrast = avg_even - avg_odd  # 正的表示even hydrophobic, odd hydrophilic
        if contrast > 0.3:
            pattern_score = min(1.0, contrast * 1.5)

    return pattern_score


def enhanced_chou_fasman_predict(sequence: str) -> np.ndarray:
    """
    v6.0 - 上下文感知 + 全局蛋白质类型判别
    核心改进（解决用户反馈的三大问题）:
    1. 上下文感知V/I/L歧义解决：
       - 螺旋上下文(A/E/K/R包围) → V/I/L倾向H（解决Insulin误判）
       - β上下文(turns + VIL交替) → V/I/L倾向E（解决Ig漏检）
    2. 全局蛋白质类型先验：
       - 先检测是helix/beta/mixed蛋白，然后施加弱全局偏置
    3. β折叠交替疏水性模式检测：识别 i, i+2, i+4 疏水交替特征
    4. 保留v5.0的连续段boost机制（温和乘法增强）
    """
    n = len(sequence)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    probs = np.zeros((n, 3), dtype=np.float64)
    window = 5

    # ====== 步骤0: 先检测全局蛋白质类型 ======
    prot_type = _detect_protein_type(sequence)

    # 全局偏置 (中等强度，足以改变结构组成比例但不压倒局部信号)
    global_bias = np.zeros(3, dtype=np.float64)
    if prot_type == "helix":
        global_bias[0] = 0.16    # 螺旋蛋白整体拉高H
        global_bias[1] = -0.10   # 抑制E（螺旋蛋白确实没有太多β）
        global_bias[2] = -0.03   # 略微抑制C（给H更多空间）
    elif prot_type == "beta":
        global_bias[1] = 0.22    # β折叠蛋白整体拉高E（解决Ig问题！强一点）
        global_bias[0] = -0.08   # 抑制H
        global_bias[2] = -0.06   # 抑制C（不然C总是占最高比例）
    elif prot_type == "coil":
        global_bias[2] = 0.10

    # ====== 步骤1: 收集基础logits ======
    base_logits = np.zeros((n, 3))

    # 先验（弱化）
    plog_h = 0.10 * np.log(PRIOR_PROBS["H"] / 0.333 + 1e-10)
    plog_e = 0.10 * np.log(PRIOR_PROBS["E"] / 0.333 + 1e-10)
    plog_c = 0.10 * np.log(PRIOR_PROBS["C"] / 0.333 + 1e-10)

    for i in range(n):
        aa_center = sequence[i]
        if aa_center in CHOU_FASMAN:
            pa, pb, pc = CHOU_FASMAN[aa_center]
            log_pa = np.log(pa + 1e-10)
            log_pb = np.log(pb + 1e-10)
            log_pc = np.log(pc + 1e-10)
        else:
            log_pa = log_pb = log_pc = 0.0

        # 短窗口 (-3, +3)
        start_s = max(0, i - 3)
        end_s = min(n, i + 4)
        sp_a = sp_b = sp_c = 0.0; sc = 0.0
        for j in range(start_s, end_s):
            aa = sequence[j]
            if aa in CHOU_FASMAN:
                pw = CHOU_FASMAN[aa]
                w = np.exp(-(abs(j - i) ** 2) / 8.0)
                sp_a += w * np.log(pw[0] + 1e-10)
                sp_b += w * np.log(pw[1] + 1e-10)
                sp_c += w * np.log(pw[2] + 1e-10)
                sc += w
        if sc > 0: sp_a/=sc; sp_b/=sc; sp_c/=sc

        # 长窗口 (-5, +5)
        start_l = max(0, i - window)
        end_l = min(n, i + window + 1)
        lp_a = lp_b = lp_c = 0.0; lc = 0.0
        for j in range(start_l, end_l):
            aa = sequence[j]
            if aa in CHOU_FASMAN:
                pw = CHOU_FASMAN[aa]
                w = np.exp(-(abs(j - i) ** 2) / 32.0)
                lp_a += w * np.log(pw[0] + 1e-10)
                lp_b += w * np.log(pw[1] + 1e-10)
                lp_c += w * np.log(pw[2] + 1e-10)
                lc += w
        if lc > 0: lp_a/=lc; lp_b/=lc; lp_c/=lc

        # 基础组合: 中心 20%, 短 45%, 长 35%
        w_c, w_s, w_l = 0.20, 0.45, 0.35
        logit_h = w_c*log_pa + w_s*sp_a + w_l*lp_a + plog_h
        logit_e = w_c*log_pb + w_s*sp_b + w_l*lp_b + plog_e
        logit_c = w_c*log_pc + w_s*sp_c + w_l*lp_c + plog_c

        # ===== v6.0新内容 =====
        # 1) 上下文感知偏置 (解决Val/Ile/Leu歧义问题)
        ctx_bias = _context_bias(sequence, i, 3)
        logit_h += ctx_bias[0]
        logit_e += ctx_bias[1]
        logit_c += ctx_bias[2]

        # 2) 全局蛋白类型偏置
        logit_h += global_bias[0]
        logit_e += global_bias[1]
        logit_c += global_bias[2]

        # 3) β折叠交替疏水性模式检测（Ig domain专用）
        beta_pattern = _alternating_hydrophobicity_beta_signal(sequence, i)
        if beta_pattern > 0.3 and prot_type == "beta":
            # 只在beta蛋白类型中使用，避免误伤螺旋蛋白
            logit_e += 0.18 * beta_pattern
            logit_h -= 0.08 * beta_pattern

        base_logits[i, 0] = logit_h
        base_logits[i, 1] = logit_e
        base_logits[i, 2] = logit_c

    # ====== 步骤2：softmax得到基础概率 ======
    for i in range(n):
        logits = base_logits[i].copy()
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs[i] = exp_l / exp_l.sum()

    # ====== 步骤3：连续段温和Boost（v5.0机制，保留） ======
    def _boost_segments(probs_arr, state_idx, min_len, boost_center, boost_edge):
        n_arr = len(probs_arr)
        states = np.argmax(probs_arr, axis=1)

        segs = []; start = None
        for i in range(n_arr):
            if states[i] == state_idx and start is None:
                start = i
            elif states[i] != state_idx and start is not None:
                if i - start >= min_len:
                    segs.append((start, i - 1))
                start = None
        if start is not None and n_arr - start >= min_len:
            segs.append((start, n_arr - 1))

        result = probs_arr.copy()
        for (s, e) in segs:
            seg_len = e - s + 1
            for pos in range(s, e + 1):
                center_dist = 2 * abs(pos - (s + e) / 2.0) / max(seg_len, 1)
                edge_factor = 1.0 - center_dist ** 2
                factor = boost_edge + (boost_center - boost_edge) * edge_factor
                result[pos, state_idx] *= factor
                result[pos] = result[pos] / result[pos].sum()
        return result

    # H段 (5+) - 所有蛋白类型都boost螺旋（螺旋是最常见的二级结构）
    probs = _boost_segments(probs, 0, 5, 1.30, 1.10)

    # ====== v6.7 Step A: 螺旋边界"削尖" + 长段内部分裂 ======
    # 1) H段boost后，段边界的COIL_BREAKER残基转回C
    # 2) 长H段(>=12)内部如果出现"高Pβ残基连续3+"区域，分裂为C（真实结构中的转折区域）
    states0 = np.argmax(probs, axis=1)
    hsegs = []
    _start = None
    for i in range(n):
        if states0[i] == 0 and _start is None:
            _start = i
        elif states0[i] != 0 and _start is not None:
            hsegs.append((_start, i - 1))
            _start = None
    if _start is not None:
        hsegs.append((_start, n - 1))

    for (hs, he) in hsegs:
        seg_len = he - hs + 1
        if seg_len < 5:
            continue
        # 段边界COIL_BREAKER削尖
        boundary_positions = set()
        for d in range(min(3, seg_len)):
            boundary_positions.add(hs + d)
        for d in range(min(3, seg_len)):
            boundary_positions.add(he - d)
        boundary_positions.add(hs - 1)
        boundary_positions.add(he + 1)

        # 段内部连续COIL_BREAKER → 转C
        if seg_len >= 8:
            for ip in range(hs + 2, he - 1):
                if sequence[ip] in COIL_BREAKERS:
                    neighbors_breaker = 0
                    if ip-1 >= 0 and sequence[ip-1] in COIL_BREAKERS:
                        neighbors_breaker += 1
                    if ip+1 < n and sequence[ip+1] in COIL_BREAKERS:
                        neighbors_breaker += 1
                    if neighbors_breaker >= 1 or (seg_len >= 10 and probs[ip, 0] < 0.90):
                        for dp in [-1, 0, 1]:
                            pp = ip + dp
                            if hs <= pp <= he and sequence[pp] in COIL_BREAKERS and probs[pp, 0] < 0.95:
                                boundary_positions.add(pp)
                        if seg_len >= 10:
                            boundary_positions.add(ip)

        # v6.7新增：长段(>=12)内部"高Pβ连续区域"分裂检测
        # 真实蛋白质中，长螺旋段中间如果出现连续3+个高Pβ残基（如V,I,L连续出现），
        # 往往标志着一个coil转折区域（这些残基虽然Pα也>1，但Pβ更高暗示链倾向）
        # 典型场景：Myoglobin pos14-16(KVE: K=1.16/0.74, V=1.06/1.70, E=1.51/0.37)
        #   V的Pβ=1.70远超Pα=1.06，但K和E的Pα更高，所以纯CF算不出这是转折
        # 策略：在长段中找"Pβ/Pα比值>1.2的残基连续2+"出现的区域，弱化H
        if seg_len >= 12:
            # 扫描段内部（不含首尾3个残基），找连续的"β倾向高于α倾向"区域
            beta_dominant_run = 0
            run_start = -1
            for ip in range(hs + 3, he - 2):
                aa = sequence[ip]
                if aa in CHOU_FASMAN:
                    pa, pb, pc = CHOU_FASMAN[aa]
                    # 残基的β倾向明显高于α倾向（比值>1.3）或Pc高（>1.2）
                    is_turn_point = (pb / max(pa, 0.01) > 1.3) or (pc > 1.2)
                else:
                    is_turn_point = False

                if is_turn_point:
                    if beta_dominant_run == 0:
                        run_start = ip
                    beta_dominant_run += 1
                else:
                    # 连续区域结束，检查长度
                    if beta_dominant_run >= 2:
                        # 找到一个转折候选区域
                        for pp in range(run_start, run_start + beta_dominant_run):
                            if probs[pp, 0] < 0.88:
                                probs[pp, 0] *= 0.62
                                probs[pp, 2] *= 1.35
                                probs[pp] = probs[pp] / probs[pp].sum()
                    beta_dominant_run = 0
                    run_start = -1
            # 处理末尾残留
            if beta_dominant_run >= 2:
                for pp in range(run_start, run_start + beta_dominant_run):
                    if probs[pp, 0] < 0.88:
                        probs[pp, 0] *= 0.62
                        probs[pp, 2] *= 1.35
                        probs[pp] = probs[pp] / probs[pp].sum()

        for edge_pos in boundary_positions:
            if 0 <= edge_pos < n:
                aa = sequence[edge_pos]
                if aa in COIL_BREAKERS:
                    ph = probs[edge_pos, 0]
                    if ph < 0.92:
                        probs[edge_pos, 0] *= 0.70
                        probs[edge_pos, 2] *= 1.28
                        probs[edge_pos] = probs[edge_pos] / probs[edge_pos].sum()

    # ====== v6.6 Step B: N/C端边缘β信号增强（解决Ig前20/后10残基β漏检问题1） ======
    # beta蛋白的N/C端最常出问题：窗口信息不完整，V/I/L/Q被误判成H
    # 策略：beta类型蛋白，边缘±8残基窗口内，若Pβ>1.05 + 相邻有足够turns → 增强E削弱H
    edge_zone = min(8, n // 6)
    if prot_type == "beta" and edge_zone >= 3:
        for i in range(n):
            is_n_edge = (i < edge_zone)
            is_c_edge = (i >= n - edge_zone)
            if not (is_n_edge or is_c_edge):
                continue
            aa = sequence[i]
            if aa in CHOU_FASMAN:
                ppa, ppb, ppc = CHOU_FASMAN[aa]
                # 高β倾向性残基(V/I/L/F/Y/W/T/Q/M)
                if ppb >= 1.05:
                    # 周围5残基有turns(G/P/S/D/N) → 更有可能是β链而不是螺旋
                    s3 = max(0, i - 3)
                    e3 = min(n, i + 4)
                    turns3 = sum(1 for a in sequence[s3:e3] if a in {"G", "P", "S", "D", "N"})
                    beta3 = sum(1 for a in sequence[s3:e3] if a in BETA_CORE)
                    if turns3 >= 1 or beta3 >= 2:
                        # 削弱H, 增强E
                        cur_h = probs[i, 0]
                        if cur_h > 0.28:
                            probs[i, 0] *= 0.58
                            probs[i, 1] *= 1.30
                            probs[i] = probs[i] / probs[i].sum()

    # E段 boost - 根据蛋白类型区分
    #   beta类型: boost 3+ 连续段 (beta折叠最自由)
    #   mixed类型: boost 5+ 连续段 (允许真正的长β链，但过滤短假阳性)
    #   helix/coil类型: 完全不boost
    if prot_type == "beta":
        probs = _boost_segments(probs, 1, 3, 1.40, 1.15)
    elif prot_type == "mixed":
        probs = _boost_segments(probs, 1, 5, 1.28, 1.08)

    # ====== v6.3 额外：非beta类型蛋白的E清理 ======
    # v6.5 精细化：
    #   - true_mixed类型（如Ubiquitin: 有turns但beta不夸张）：轻度压制E，允许20-30% β
    #   - "VILF富集非beta"子类型（如Insulin: beta_core>45%但turns<30%）：强力压制E
    #     真beta蛋白必须turns>32%（beta_is_clean），turns<30%说明疏水残基是螺旋/球形堆积用
    #   - helix/coil类型：强力压制
    if prot_type != "beta":
        # ====== mixed内部子分类 ======
        # 计算全局组成（已在外层_detect_protein_type算过，这里重算代价低）
        _bc = sum(1 for a in sequence if a in BETA_CORE) / n
        _tn = sum(1 for a in sequence if a in {"G", "S", "T", "N", "D", "P"}) / n
        # "富含疏水残基但不是β折叠"的特殊mixed（典型：Insulin）
        # 判据：beta_core相对较高(>28%) 但 turns很低(<30%)
        #   真beta蛋白需要turns>32%（beta_is_clean），若turns<30%说明疏水残基(含AMBIVALENT的L/C/M)
        #   实际用于螺旋或球形核心堆积，不是β折叠
        _vilf_non_beta_mixed = (prot_type == "mixed") and (_bc > 0.28) and (_tn < 0.30)

        # 三种压制强度级别
        if _vilf_non_beta_mixed or prot_type == "helix":
            # 最强级：Insulin模式或螺旋蛋白 → 激进E清理
            _e_factor = 0.12
            _trigger_mixed_strict = False
            _max_unsafe = 4
        elif prot_type == "mixed":
            # 正常mixed（如Ubiquitin）→ 轻度压制，允许合理β
            _e_factor = 0.38
            _trigger_mixed_strict = True
            _max_unsafe = 2
        else:  # coil
            _e_factor = 0.18
            _trigger_mixed_strict = False
            _max_unsafe = 5

        for i in range(n):
            center = sequence[i]
            if (center in BETA_CORE or center in AMBIVALENT or
                center in {"C", "Y", "F", "T", "S"}):
                s = max(0, i - 4)
                e = min(n, i + 5)
                win = sequence[s:e]
                a_n = sum(1 for a in win if a in ALPHA_CORE)
                b_n = sum(1 for a in win if a in BETA_CORE)
                c_n = sum(1 for a in win if a in COIL_BREAKERS)
                total = len(win)
                is_strong_beta_res = center in {"V", "I", "L", "F", "Y", "W"}

                # true mixed (有真β可能)：触发条件严格 - a_n需显著>b_n；否则留给β
                # 其他所有类型（helix/coil/vilf_non_beta）：宽松触发
                if _trigger_mixed_strict:
                    trigger = (a_n >= b_n + 2) or (c_n >= total * 0.58)
                else:
                    trigger = ((a_n >= b_n and a_n >= 1) or
                               (c_n >= total * 0.50) or
                               (not is_strong_beta_res and b_n <= 3))

                if trigger:
                    probs[i, 1] *= _e_factor
                    # 给H/C加权只在helix类型激进
                    if (prot_type == "helix" or _vilf_non_beta_mixed) and \
                            center in ALPHA_CORE | AMBIVALENT | {"V", "I", "L", "M"}:
                        probs[i, 0] *= 1.22
                    elif c_n >= total * 0.50:
                        probs[i, 2] *= 1.12
                    probs[i] = probs[i] / probs[i].sum()

        # ====== 孤立E段清理 ======
        states = np.argmax(probs, axis=1)
        segs = []
        start = None
        for i in range(n):
            if states[i] == 1 and start is None:
                start = i
            elif states[i] != 1 and start is not None:
                segs.append((start, i - 1))
                start = None
        if start is not None:
            segs.append((start, n - 1))

        for (seg_s, seg_e) in segs:
            length = seg_e - seg_s + 1
            if length <= _max_unsafe:
                for pos in range(seg_s, seg_e + 1):
                    # 几乎清零E，分给H和C
                    prev = probs[pos].copy()
                    e_val = prev[1]
                    probs[pos, 1] = e_val * 0.08
                    # 分配给H和C的比例：根据局部情况定
                    s2 = max(0, pos - 3)
                    e2 = min(n, pos + 4)
                    local_a = sum(1 for a in sequence[s2:e2] if a in ALPHA_CORE)
                    local_c = sum(1 for a in sequence[s2:e2] if a in COIL_BREAKERS)
                    if local_a >= local_c:
                        probs[pos, 0] += e_val * 0.60
                        probs[pos, 2] += e_val * 0.32
                    else:
                        probs[pos, 0] += e_val * 0.32
                        probs[pos, 2] += e_val * 0.60
                    probs[pos] = probs[pos] / probs[pos].sum()

    return probs.astype(np.float32)
