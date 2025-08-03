from pyscipopt import Model

# 모델 생성
model = Model("LP_Problem")

# 변수 정의
d1 = model.addVar(vtype="C", name="d1", ub=4, obj=-3)  # d1 ≤ 4
d2 = model.addVar(vtype="C", name="d2", ub=6, obj=-6)  # d2 ≤ 6
p1 = model.addVar(vtype="C", name="p1", ub=5, obj=1.0)  # p1 ≤ 5
p2 = model.addVar(vtype="C", name="p2", ub=5, obj=4.0)  # p2 ≤ 5
cap = 4.0
f12 = model.addVar(vtype="C", name="f12", lb=-cap, ub=cap)  # -4 ≤ f₁₋₂ ≤ 4



# 제약조건 추가
# Constraint (10): p1 - f12 = d1
model.addCons(p1 - f12 == d1, "flow_balance_1")

# Constraint (11): p2 + f12 = d2  
model.addCons(p2 + f12 == d2, "flow_balance_2")

# 모델 최적화

from pyscipopt import SCIP_PARAMSETTING
model.setPresolve(SCIP_PARAMSETTING.OFF)
model.setHeuristics(SCIP_PARAMSETTING.OFF)
model.disablePropagation()

model.optimize()

# 결과 출력
if model.getStatus() == "optimal":
    print("최적해를 찾았습니다!")
    print(f"목적함수 값: {model.getObjVal():.4f}")
    print("\n변수 값들:")
    print(f"d1 = {model.getVal(d1):.4f}")
    print(f"d2 = {model.getVal(d2):.4f}")
    print(f"p1 = {model.getVal(p1):.4f}")
    print(f"p2 = {model.getVal(p2):.4f}")
    print(f"f12 = {model.getVal(f12):.4f}")
    
    # 제약조건 검증
    print("\n제약조건 검증:")
    p1_val = model.getVal(p1)
    p2_val = model.getVal(p2)
    d1_val = model.getVal(d1)
    d2_val = model.getVal(d2)
    f12_val = model.getVal(f12)
    
    print(f"p1 - f12 = {p1_val:.4f} - {f12_val:.4f}, d1 = {d1_val:.4f}")
    print(f"p2 + f12 = {p2_val:.4f} + {f12_val:.4f}, d2 = {d2_val:.4f}")
    
    # 듀얼 변수 (그림자 가격) 출력
    print("\n듀얼 변수 (그림자 가격):")
    constraints = model.getConss(True)
    for cons in constraints:
        dual_val = model.getDualsolLinear(cons)
        print(f"{cons.name}: λ = {dual_val:.4f}")
        
else:
    print(f"최적화 실패. 상태: {model.getStatus()}")

# 모델 정보 출력
print(f"\n모델 통계:")
print(f"변수 개수: {model.getNVars()}")
print(f"제약조건 개수: {model.getNConss()}")