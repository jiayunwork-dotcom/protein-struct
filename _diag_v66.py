
print("="*70)
print("验证用户反馈的三个问题 v6.6")
print("="*70)
from src.prediction import GOR4Predictor, NNPredictor, LSTMPredictor
import numpy as np
gor,nn,lstm = GOR4Predictor(),NNPredictor(),LSTMPredictor()

# ===== 问题1: Myoglobin KVE(14-16)和GCCC(28-31) =====
mb = "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMK"
mb_true = "CCHHHHHHHHHHHHHCCCHHHHHHHHHHHCCCCHHHHHHHHHHHCCCHHHHHHHHHH"[:57]
print("\n【问题1】Myoglobin 螺旋边界")
print(f"真实: {mb_true}")
print(f"GOR : {''.join(gor.predict(mb).states)}")
print(f"NN  : {''.join(nn.predict(mb).states)}")
print(f"LSTM: {''.join(lstm.predict(mb).states)}")
# 标记pos14-16
print(f"      {' '*13}↑↑↑ pos14-16 (KVE真实=CCC)")
print(f"      {' '*27}↑↑↑↑ pos28-31 (GQDI真实=CCCC，用户说GCCC)")
# 精确看12-18和26-33
for i,(t,g,n,l) in enumerate(zip(mb_true,gor.predict(mb).states,nn.predict(mb).states,lstm.predict(mb).states)):
    if 12 <= i <= 33:
        print(f"  pos{i:2d}({mb[i]}): true={t} GOR={g} NN={n} LSTM={l}  {'❌' if g!=t or n!=t or l!=t else '✅'}")

# ===== 问题2: Ig前20残基GOR E识别 =====
ig = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGST"
print("\n【问题2】Ig前20残基 β折叠识别")
for i in range(20):
    t = "CEEEECCEEEEEECCCCCEEEE"[:58][i] if i < 58 else "?"
    g = gor.predict(ig).states[i]
    n = nn.predict(ig).states[i]
    l = lstm.predict(ig).states[i]
    print(f"  pos{i:2d}({ig[i]}): GOR={g} NN={n} LSTM={l}  E? GOR={'✅'if g=='E'else'❌'} NN={'✅'if n=='E'else'❌'} LSTM={'✅'if l=='E'else'❌'}")
e_gor = sum(1 for i in range(20) if gor.predict(ig).states[i]=='E')
e_nn = sum(1 for i in range(20) if nn.predict(ig).states[i]=='E')
e_lstm = sum(1 for i in range(20) if lstm.predict(ig).states[i]=='E')
print(f"前20 E总数: GOR={e_gor}, NN={e_nn}, LSTM={e_lstm} (目标≥8)")

# ===== 问题3: NN/LSTM差异 =====
print("\n【问题3】NN vs LSTM 差异数量")
tests = [("Ig",ig,58),("Myoglobin",mb,57),("Insulin","GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKA",51),
         ("Ubiquitin","MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTL",56)]
for name,s,L in tests:
    sn = "".join(nn.predict(s).states[:L])
    sl = "".join(lstm.predict(s).states[:L])
    diffs = sum(1 for a,b in zip(sn,sl) if a!=b)
    print(f"  {name:12s}: {diffs:2d}处差异 (len={L}, 比例{diffs/L*100:.1f}%)")
print("目标: ≥10处差异/50aa (约20%+)")

print("\n"+"="*70)
