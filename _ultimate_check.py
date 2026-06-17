print("=" * 78)
print("v6.6 FINAL 验证 - 全部6个问题（3新+3旧）")
print("=" * 78)

from src.prediction import GOR4Predictor, NNPredictor, LSTMPredictor
import numpy as np

gor, nn, lstm = GOR4Predictor(), NNPredictor(), LSTMPredictor()

# 新1: Ig前20E
print("\n■ 新问题1: Ig前20残基β折叠 (目标60%=12个E)")
seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGST"
true = "CEEEECCEEEEECCCCCEEEECCCEEEEECCCCEECCCCEEEEECCCEEEECCCEE"[:58]
for nm, r in [("GOR",gor.predict(seq)),("NN",nn.predict(seq)),("LSTM",lstm.predict(seq))]:
    p = "".join(r.states)
    e20 = p[:20].count('E'); t20 = true[:20].count('E')
    c20 = sum(1 for a,b in zip(p[:20],true[:20]) if a==b)
    mark = "✅" if e20 >= 0.5*t20 else "⚠️"
    print(f"  {nm:5s}: 前20 E={e20}/{t20} ({c20}/20正确) {mark}  |  全E={p.count('E')}/58")

# 新2: NN/LSTM差异
print("\n■ 新问题2: NN/LSTM差异性（≥8处差异用户可区分）")
tests = [("Ig",seq),("Myo","MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKI"),("Ins","GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKA"),("Ubi","MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTL")]
diff_cnt = 0
for nm, s in tests:
    sn = "".join(nn.predict(s).states); sl = "".join(lstm.predict(s).states)
    sg = "".join(gor.predict(s).states)
    d = sum(1 for a,b in zip(sn,sl) if a!=b)
    diff_cnt += d
    m = "✅" if d>=5 else "❌"
    print(f"  {nm:4s}: NN≠LSTM 差{d:2d}处 {m}  |  NN≠GOR={sum(1 for a,b in zip(sn,sg) if a!=b):2d}, LSTM≠GOR={sum(1 for a,b in zip(sl,sg) if a!=b):2d}")
avg = diff_cnt/len(tests)
m_avg = "✅" if avg >= 8 else "⚠️"
print(f"  >>> 平均差异: {avg:.1f}处 (目标≥8) {m_avg}")

# 新3: Myoglobin边界
print("\n■ 新问题3: Myoglobin螺旋边界（H不要过度预测）")
myo = "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMK"
true_m = "CCHHHHHHHHHHHHHCCCHHHHHHHHHHHCCCCHHHHHHHHHHHCCCHHHHHHHHHH"
ht = true_m.count('H'); ct = true_m.count('C')
bounds = [(14,16),(28,31),(46,48)]
for nm, r in [("GOR",gor.predict(myo)),("NN",nn.predict(myo)),("LSTM",lstm.predict(myo))]:
    p = "".join(r.states)
    hp = p.count('H')
    issues = []
    for (s,e) in bounds:
        w = sum(1 for i in range(s,e+1) if true_m[i]!=p[i])
        if w: issues.append(f"p{s}-{e}:真{true_m[s:e+1]}→预{p[s:e+1]}({w}错)")
    m1 = "✅" if hp-ht <= 5 else "❌"
    m2 = "✅" if len(issues) <= 2 else "⚠️"
    print(f"  {nm:5s}: H真{ht}→预{hp}(+{hp-ht:+d}) {m1}  |  边界: {'; '.join(issues) if issues else '无'} {m2}")
    print(f"         真:{true_m[:57]}\n         预:{p[:57]}")

# 旧问题回归
print("\n" + "="*78 + "\n■ 旧问题回归（3个核心不能回退）\n" + "="*78)
ig_e_tar = 31; ins_e_tar = 0; myo_h_tar = 45
for nm, r in [("Ig-GOR",rg:=gor.predict(seq)),("Ig-NN",rn:=nn.predict(seq)),("Ig-LSTM",rl:=lstm.predict(seq))]:
    e = "".join(r.states).count('E')
    m = "✅" if e >= int(ig_e_tar*0.58) else "❌"
    print(f"  {nm:8s}: E={e:2d}/58 (目标≈{ig_e_tar}, ≥{int(ig_e_tar*0.58)}合格) {m}")
ins = "GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKA"
for nm, r in [("Ins-GOR",gor.predict(ins)),("Ins-NN",nn.predict(ins)),("Ins-LSTM",lstm.predict(ins))]:
    e = "".join(r.states).count('E')
    m = "✅" if e <= 3 else "❌"
    print(f"  {nm:8s}: E={e:2d}/51 (目标0, ≤3合格) {m}")

# 综合PDB总体
print("\n" + "="*78 + "\n■ 5个PDB蛋白Q3综合（test_predictions）\n" + "="*78)
from test_predictions import test_data
from src.evaluation import compute_q3
import math
q3s = {"GOR-IV":[],"Neural Network":[],"Bidirectional LSTM":[]}
predictors = {"GOR-IV": gor, "Neural Network": nn, "Bidirectional LSTM": lstm}
for t in test_data:
    n = min(len(t["sequence"]),len(t["actual"]))
    s = t["sequence"][:n]; a = t["actual"][:n]
    for pname, pred in predictors.items():
        q3s[pname].append(compute_q3("".join(pred.predict(s).states), a))

for pname, arr in q3s.items():
    m = np.mean(arr)*100; s = np.std(arr)*100
    print(f"  {pname:20s}: Q3={m:5.1f}% ±{s:4.1f}%  (5个案例: "+" ,".join(f"{x*100:4.1f}%" for x in arr)+")")

print("\n" + "="*78)
print("全部验证完成")
print("="*78)
