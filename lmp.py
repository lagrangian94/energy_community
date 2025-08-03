import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

# =============================================================================
# 데이터 정의 부분 (외부에서 정의)
# =============================================================================

# 지역 정의
regions = [
    'hokkaido', 'tohoku', 'tokyo', 'chubu', 'hokuriku',
    'kansai', 'chugoku', 'shikoku', 'kyushu'
]

# 지역별 수요 (MW)
demand_data = {
    'hokkaido': 4160,
    'tohoku': 13380, 
    'tokyo': 54990,
    'chubu': 24550,
    'hokuriku': 4950,   
    'kansai': 27410,
    'chugoku': 10430,
    'shikoku': 4970,
    'kyushu': 15370,
}

# 발전기 데이터: {region: {gen_id: {cost, capacity, type}}}
generator_data = {
    'hokkaido': {  # A - 수요: 4,160 MW
        'A_1': {'cost': 7, 'capacity': 2000, 'type': 'coal'},
        'A_2': {'cost': 8, 'capacity': 1500, 'type': 'coal'},
        'A_3': {'cost': 12, 'capacity': 1200, 'type': 'gas'},
        'A_4': {'cost': 3, 'capacity': 800, 'type': 'hydro'},
        'A_5': {'cost': 5, 'capacity': 1000, 'type': 'wind'},
    },
    'tohoku': {  # B - 수요: 13,380 MW
        'B_1': {'cost': 6, 'capacity': 4000, 'type': 'nuclear'},
        'B_2': {'cost': 9, 'capacity': 3000, 'type': 'coal'},
        'B_3': {'cost': 10, 'capacity': 2500, 'type': 'coal'},
        'B_4': {'cost': 13, 'capacity': 2000, 'type': 'gas'},
        'B_5': {'cost': 4, 'capacity': 1500, 'type': 'hydro'},
        'B_6': {'cost': 6, 'capacity': 1800, 'type': 'wind'},
        'B_7': {'cost': 8, 'capacity': 500, 'type': 'geothermal'},
    },
    'tokyo': {  # C - 수요: 54,990 MW
        'C_1': {'cost': 11, 'capacity': 8000, 'type': 'gas'},
        'C_2': {'cost': 12, 'capacity': 7500, 'type': 'gas'},
        'C_3': {'cost': 13, 'capacity': 7000, 'type': 'gas'},
        'C_4': {'cost': 10, 'capacity': 6000, 'type': 'coal'},
        'C_5': {'cost': 15, 'capacity': 4000, 'type': 'oil'},
        'C_6': {'cost': 5, 'capacity': 3500, 'type': 'hydro'},
        'C_7': {'cost': 8, 'capacity': 3000, 'type': 'solar'},
        'C_8': {'cost': 14, 'capacity': 5000, 'type': 'gas'},
        'C_9': {'cost': 16, 'capacity': 3000, 'type': 'oil'},
        'C_10': {'cost': 9, 'capacity': 4000, 'type': 'coal'},
    },
    'chubu': {  # D - 수요: 24,550 MW
        'D_1': {'cost': 5, 'capacity': 4500, 'type': 'nuclear'},
        'D_2': {'cost': 8, 'capacity': 3500, 'type': 'coal'},
        'D_3': {'cost': 11, 'capacity': 4000, 'type': 'gas'},
        'D_4': {'cost': 12, 'capacity': 3000, 'type': 'gas'},
        'D_5': {'cost': 4, 'capacity': 2000, 'type': 'hydro'},
        'D_6': {'cost': 7, 'capacity': 1500, 'type': 'solar'},
        'D_7': {'cost': 9, 'capacity': 2500, 'type': 'coal'},
        'D_8': {'cost': 13, 'capacity': 2000, 'type': 'gas'},
        'D_9': {'cost': 14, 'capacity': 1500, 'type': 'oil'}
    },
    'hokuriku': {  # E - 수요: 4,950 MW
        'E_1': {'cost': 6, 'capacity': 2500, 'type': 'nuclear'},
        'E_2': {'cost': 3, 'capacity': 2000, 'type': 'hydro'},
        'E_3': {'cost': 9, 'capacity': 1800, 'type': 'coal'},
        'E_4': {'cost': 13, 'capacity': 1200, 'type': 'gas'},
        'E_5': {'cost': 6, 'capacity': 800, 'type': 'wind'},
    },
    'kansai': {  # F - 수요: 27,410 MW
        'F_1': {'cost': 5, 'capacity': 6000, 'type': 'nuclear'},
        'F_2': {'cost': 6, 'capacity': 5000, 'type': 'nuclear'},
        'F_3': {'cost': 8, 'capacity': 4500, 'type': 'coal'},
        'F_4': {'cost': 11, 'capacity': 3500, 'type': 'gas'},
        'F_5': {'cost': 4, 'capacity': 2500, 'type': 'hydro'},
        'F_6': {'cost': 7, 'capacity': 2000, 'type': 'solar'},
        'F_7': {'cost': 9, 'capacity': 3000, 'type': 'coal'},
        'F_8': {'cost': 12, 'capacity': 2500, 'type': 'gas'},
    },
    'chugoku': {  # G - 수요: 10,430 MW
        'G_1': {'cost': 6, 'capacity': 3500, 'type': 'nuclear'},
        'G_2': {'cost': 10, 'capacity': 2500, 'type': 'coal'},
        'G_3': {'cost': 12, 'capacity': 2000, 'type': 'gas'},
        'G_4': {'cost': 16, 'capacity': 1500, 'type': 'oil'},
        'G_5': {'cost': 4, 'capacity': 1200, 'type': 'hydro'},
        'G_6': {'cost': 8, 'capacity': 1000, 'type': 'solar'},
        'G_7': {'cost': 11, 'capacity': 1800, 'type': 'coal'},
    },
    'shikoku': {  # H - 수요: 4,970 MW
        'H_1': {'cost': 7, 'capacity': 2500, 'type': 'nuclear'},
        'H_2': {'cost': 12, 'capacity': 2000, 'type': 'coal'},
        'H_3': {'cost': 14, 'capacity': 1500, 'type': 'gas'},
        'H_4': {'cost': 17, 'capacity': 1000, 'type': 'oil'},
        'H_5': {'cost': 5, 'capacity': 1200, 'type': 'hydro'},
        'H_6': {'cost': 9, 'capacity': 800, 'type': 'solar'},
    },
    'kyushu': {  # I - 수요: 15,370 MW
        'I_1': {'cost': 6, 'capacity': 4000, 'type': 'nuclear'},
        'I_2': {'cost': 9, 'capacity': 3500, 'type': 'coal'},
        'I_3': {'cost': 11, 'capacity': 2500, 'type': 'gas'},
        'I_4': {'cost': 15, 'capacity': 1500, 'type': 'oil'},
        'I_5': {'cost': 4, 'capacity': 2000, 'type': 'hydro'},
        'I_6': {'cost': 7, 'capacity': 1200, 'type': 'geothermal'},
        'I_7': {'cost': 8, 'capacity': 2500, 'type': 'solar'},
        'I_8': {'cost': 6, 'capacity': 1500, 'type': 'wind'},
    },
}

# 송전선로 연결 정의
connections = [
    ('hokkaido', 'tohoku'), ('tohoku', 'hokkaido'),
    ('tohoku', 'tokyo'), ('tokyo', 'tohoku'),
    ('tokyo', 'chubu'), ('chubu', 'tokyo'),
    ('chubu', 'hokuriku'), ('hokuriku', 'chubu'),
    ('chubu', 'kansai'), ('kansai', 'chubu'),
    ('hokuriku', 'kansai'), ('kansai', 'hokuriku'), 
    ('kansai', 'chugoku'), ('chugoku', 'kansai'),
    ('chugoku', 'shikoku'), ('shikoku', 'chugoku'),
    ('chugoku', 'kyushu'), ('kyushu', 'chugoku'),
    ('shikoku', 'kansai'),
]

# 송전선로 용량 데이터
fmax_data = {
    'hokkaido': {'tohoku': 900},
    'tohoku': {'hokkaido': 900, 'tokyo': 5760},
    'tokyo': {'tohoku': 2360, 'chubu': 2100},
    'chubu': {'tokyo': 2100, 'hokuriku': 300, 'kansai': 1160},
    'hokuriku': {'chubu': 300, 'kansai': 1900},
    'kansai': {'hokuriku': 1500, 'chubu': 2500, 'chugoku': 2780, 'shikoku': 0},
    'chugoku': {'kansai': 4650, 'shikoku': 1200, 'kyushu': 210},
    'shikoku': {'kansai': 1400, 'chugoku': 1200},
    'kyushu': {'chugoku': 2410}
}

# =============================================================================
# 모델 생성 함수
# =============================================================================
def generate_lmp_model():
    """LMP 모델을 생성하는 함수"""
    print("=== 일본 전력시스템 다중발전기 LMP 모델 생성 ===")
    
    # 모델 생성
    model = pyo.ConcreteModel()
    
    # 집합 정의
    model.N = pyo.Set(initialize=regions, doc="Japanese power system regions")
    
    # 모든 발전기 ID 집합 생성
    all_generators = []
    for region in generator_data:
        for gen_id in generator_data[region]:
            all_generators.append(gen_id)
    model.G = pyo.Set(initialize=all_generators, doc="All generators")
    
    # 지역별 발전기 집합 G_i 생성
    def get_region_generators(model, i):
        """지역 i의 발전기 집합 G_i"""
        return [gen_id for gen_id in generator_data.get(i, {})]
    model.G_i = pyo.Set(model.N, initialize=get_region_generators, doc="Generators in each region")
    
    # 송전선로 집합
    model.L = pyo.Set(initialize=connections, doc="Transmission line connections")
    
    # 파라미터 정의
    model.d = pyo.Param(model.N, initialize=demand_data, doc="Demand at each region (MW)")
    
    # 발전기별 비용 계수
    def gen_cost_init(model, g):
        for region in generator_data:
            if g in generator_data[region]:
                return generator_data[region][g]['cost']
        return 0
    model.c_g = pyo.Param(model.G, initialize=gen_cost_init, doc="Generation cost by generator (¥/MWh)")
    
    # 발전기별 용량
    def gen_capacity_init(model, g):
        for region in generator_data:
            if g in generator_data[region]:
                return generator_data[region][g]['capacity']
        return 0
    model.p_g_max = pyo.Param(model.G, initialize=gen_capacity_init, doc="Generator capacity (MW)")
    
    # 송전선로 용량 초기화 함수
    def fmax_init(model, i, j):
        """송전선로 용량 초기화 함수"""
        return fmax_data.get(i, {}).get(j, 0)
    model.f_max = pyo.Param(model.L, initialize=fmax_init, doc="Line capacity limits (MW)")
    
    # 변수 정의
    model.p_g = pyo.Var(model.G, bounds=lambda model, g: (0, model.p_g_max[g]), 
                         doc="Generation by each generator (MW)")
    model.f = pyo.Var(model.L, bounds=lambda model, i, j: (0, model.f_max[i,j]),
                       doc="Power flow on transmission lines (MW)")
    
    # 듀얼 변수를 위한 suffix
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    # 목적함수 정의
    model.obj = pyo.Objective(
        expr=sum(model.c_g[g] * model.p_g[g] for g in model.G),
        sense=pyo.minimize,
        doc="Minimize total generation cost by all generators (¥/h)"
    )
    
    # 제약조건 정의
    def power_balance_rule(model, i):
        """지역 i에서의 전력 수급 균형"""
        total_generation = sum(model.p_g[g] for g in model.G_i[i])
        inflow = sum(model.f[j,i] for j in model.N if (j,i) in model.L)
        outflow = sum(model.f[i,j] for j in model.N if (i,j) in model.L)
        net_outflow = outflow - inflow
        return total_generation - model.d[i] - net_outflow == 0
    
    model.power_balance = pyo.Constraint(model.N, rule=power_balance_rule, 
                                        doc="Power balance at each region")
    
    # Feasibility 체크
    print(f"\n=== Feasibility 사전 체크 ===")
    print("지역별 자체 발전 용량 vs 수요:")
    for i in model.N:
        region_capacity = sum(model.p_g_max[g] for g in model.G_i[i])
        region_demand = model.d[i]
        ratio = region_capacity / region_demand
        status = "OK" if ratio >= 1.0 else "부족"
        print(f"  {i:10s}: 용량 {region_capacity:5,} MW / 수요 {region_demand:5,} MW = {ratio:5.2f} ({status})")
    
    print(f"\n총 용량 vs 총 수요:")
    total_capacity = sum(model.p_g_max[g] for g in model.G)
    total_demand = sum(model.d[i] for i in model.N)
    print(f"  총 용량: {total_capacity:6,} MW")
    print(f"  총 수요: {total_demand:6,} MW")
    print(f"  비율:   {total_capacity/total_demand:6.2f}")
    
    # 모델 정보 출력
    print(f"\n=== 모델 정보 ===")
    print(f"지역 수: {len(model.N)}")
    print(f"총 발전기 수: {len(model.G)}")
    print(f"송전선로 수: {len(model.L)}")
    print(f"총 수요: {sum(model.d[i] for i in model.N):,} MW")
    
    # 수리모형 출력
    print(f"\n=== 수리모형 (Mathematical Formulation) ===")
    print("다중발전기 모델: 발전기별 비용 최소화")
    print("")
    print("minimize  z = Σ   Σ   c_g × p_g")
    print("            i∈N g∈G_i")
    print("")
    print("subject to:")
    print("")
    print("  [전력 수급균형]")
    print("    Σ   p_g - d_i -  Σ   f_i,j = 0  : λ_i    ∀i ∈ N")
    print("  g∈G_i            j∈Ω_i")
    print("")
    print("  [발전기 용량 제약]") 
    print("  0 ≤ p_g ≤ p_g^max                        ∀g ∈ G")
    print("")
    print("  [송전 용량 제약]")
    print("  0 ≤ f_i,j ≤ f_i,j^max                    ∀(i,j) ∈ L")
    print("")
    print("where:")
    print("  N     = {hokkaido, tohoku, tokyo, chubu, hokuriku, kansai, chugoku, shikoku, kyushu}")
    print("  G     = {모든 발전기 집합}")
    print("  G_i   = {지역 i의 발전기 집합}")
    print("  L     = {송전선로 집합}")
    print("  c_g   = 발전기 g의 비용 계수 (¥/MWh)")
    print("  p_g   = 발전기 g의 발전량 (MW)")
    print("  d_i   = 지역 i의 수요 (MW, 상수)")
    print("  λ_i   = 지역 i의 LMP (¥/MWh, 듀얼 변수)")
    
    # 지역별 발전기 정보
    print(f"\n지역별 발전기 정보:")
    for i in model.N:
        generators_in_region = list(model.G_i[i])
        total_capacity = sum(model.p_g_max[g] for g in generators_in_region)
        avg_cost = sum(model.c_g[g] for g in generators_in_region) / len(generators_in_region) if generators_in_region else 0
        
        print(f"\n  지역 {i}:")
        print(f"    수요: {model.d[i]:5,} MW")
        print(f"    발전기 수: {len(generators_in_region)}개") 
        print(f"    총 발전용량: {total_capacity:5,} MW")
        print(f"    평균 비용: {avg_cost:4.1f} ¥/MWh")
        
        # 발전기 타입별로 분류해서 출력
        gen_by_type = {}
        for g in generators_in_region:
            gen_type = None
            for region in generator_data:
                if g in generator_data[region]:
                    gen_type = generator_data[region][g]['type']
                    break
            if gen_type not in gen_by_type:
                gen_by_type[gen_type] = []
            gen_by_type[gen_type].append(g)
        
        type_summary = []
        for gen_type, gens in gen_by_type.items():
            type_capacity = sum(model.p_g_max[g] for g in gens)
            type_summary.append(f"{gen_type}({len(gens)}기, {type_capacity}MW)")
        
        print(f"    G_{i} = {{{', '.join(type_summary)}}}")
    
    print(f"\n모델 크기:")
    print(f"  변수 수: {len(model.G)} (발전기) + {len(model.L)} (송전) = {len(model.G) + len(model.L)}")
    print(f"  제약조건 수: {len(model.N)} (수급균형) + {len(model.G)} (발전기용량) + {len(model.L)} (송전용량)")
    print(f"               = {len(model.N) + len(model.G) + len(model.L)}")
    
    return model

# =============================================================================
# 최적화 함수
# =============================================================================
def solve_lmp_model(model):
    """LMP 최적화 모델을 실행하고 결과를 출력하는 함수"""
    print("\n=== 최적화 실행 ===")
    
    # 솔버 설정
    solver = pyo.SolverFactory('gurobi')
    if not solver.available():
        print("Gurobi 솔버를 사용할 수 없습니다.")
        solver = pyo.SolverFactory('glpk')
        if solver.available():
            print("GLPK 솔버를 사용합니다.")
        else:
            print("사용 가능한 솔버가 없습니다.")
            return None
    
    # 최적화 실행
    results = solver.solve(model, tee=False)
    
    # 결과 확인
    if (results.solver.status == SolverStatus.ok) and \
       (results.solver.termination_condition == TerminationCondition.optimal):
        
        print("\n=== 최적해 (Optimal Solution) ===")
        print(f"목적함수 값: z* = {pyo.value(model.obj):,.0f} ¥/h")
        
        print(f"\n지역별 발전 결과:")
        total_gen = 0
        total_cost = 0
        
        for i in model.N:
            print(f"\n  지역 {i} (수요: {model.d[i]:,} MW):")
            
            region_generation = 0
            region_cost = 0
            
            # 발전기별 결과를 타입별로 분류
            gen_by_type = {}
            for g in model.G_i[i]:
                p_g_val = pyo.value(model.p_g[g])
                if p_g_val > 1e-6:  # 활성 발전기만
                    # 발전기 타입 찾기
                    gen_type = None
                    for region in generator_data:
                        if g in generator_data[region]:
                            gen_type = generator_data[region][g]['type']
                            break
                    
                    if gen_type not in gen_by_type:
                        gen_by_type[gen_type] = []
                    gen_by_type[gen_type].append((g, p_g_val, model.c_g[g]))
                    
                    region_generation += p_g_val
                    region_cost += p_g_val * model.c_g[g]
            
            # 타입별로 정렬해서 출력
            for gen_type in sorted(gen_by_type.keys()):
                gens_data = gen_by_type[gen_type]
                type_gen = sum(p for _, p, _ in gens_data)
                print(f"    [{gen_type.upper()}] {type_gen:6,.0f} MW:")
                
                # 비용 순으로 정렬
                for g, p_g_val, cost in sorted(gens_data, key=lambda x: x[2]):
                    utilization = (p_g_val / model.p_g_max[g]) * 100
                    gen_cost = p_g_val * cost
                    print(f"      {g}: {p_g_val:5,.0f} MW ({utilization:5.1f}%), 비용: {gen_cost:8,.0f} ¥/h")
            
            print(f"    총 발전량: {region_generation:6,.0f} MW")
            print(f"    총 비용:   {region_cost:10,.0f} ¥/h")
            print(f"    평균 비용: {region_cost/region_generation:6.2f} ¥/MWh" if region_generation > 0 else "    평균 비용: N/A")
            
            total_gen += region_generation
            total_cost += region_cost
        
        print(f"\n=== LMP (λi* - Locational Marginal Price) ===")
        lmp_values = {}
        try:
            for i in model.N:
                if model.power_balance[i] in model.dual:
                    lmp = model.dual[model.power_balance[i]]
                    lmp_values[i] = lmp
                    print(f"  λ_{i}* = {lmp:8.2f} ¥/MWh")
                else:
                    print(f"  λ_{i}*: 듀얽 값 없음")
        except Exception as e:
            print(f"듀얼 변수 정보를 가져올 수 없습니다: {e}")
        
        print(f"\n=== 시스템 요약 ===")
        print(f"총 발전량: {total_gen:8,.0f} MW")
        print(f"총 수요:   {sum(model.d[i] for i in model.N):8,} MW") 
        print(f"총 비용:   {total_cost:12,.0f} ¥/h")
        print(f"평균 비용: {total_cost/total_gen:8.2f} ¥/MWh")
        
        # Merit Order 분석
        active_gens = [(g, model.c_g[g], pyo.value(model.p_g[g])) for g in model.G 
                       if pyo.value(model.p_g[g]) > 1e-6]
        if active_gens:
            active_gens.sort(key=lambda x: x[1])  # 비용 순 정렬
            print(f"\nMerit Order (경제급전 순서 - 상위 10개):")
            for idx, (g, cost, generation) in enumerate(active_gens[:10], 1):
                # 지역 찾기
                region = None
                gen_type = None
                for r in generator_data:
                    if g in generator_data[r]:
                        region = r
                        gen_type = generator_data[r][g]['type']
                        break
                print(f"  {idx:2}. {g} ({region}, {gen_type}): {cost} ¥/MWh, {generation:6,.0f} MW")
        
        return results
    
    else:
        print(f"\n최적해를 찾지 못했습니다.")
        print(f"솔버 상태: {results.solver.status}")
        print(f"종료 조건: {results.solver.termination_condition}")
        return results

# =============================================================================
# Infeasibility 진단 함수  
# =============================================================================
def diagnose_infeasibility():
    """Infeasibility 진단을 위한 slack variable 모델을 실행하는 함수"""
    print("\n=== Infeasibility 진단 모델 생성 ===")

    # 새로운 진단용 모델 생성
    diag_model = pyo.ConcreteModel()
    
    # 집합 정의
    diag_model.N = pyo.Set(initialize=regions, doc="Regions")
    
    all_generators = []
    for region in generator_data:
        for gen_id in generator_data[region]:
            all_generators.append(gen_id)
    diag_model.G = pyo.Set(initialize=all_generators, doc="All generators")
    
    def get_region_generators(model, i):
        return [gen_id for gen_id in generator_data.get(i, {})]
    diag_model.G_i = pyo.Set(diag_model.N, initialize=get_region_generators, doc="Generators in each region")
    diag_model.L = pyo.Set(initialize=connections, doc="Transmission lines")

    # 파라미터들
    diag_model.d = pyo.Param(diag_model.N, initialize=demand_data, doc="Demand")
    
    def gen_cost_init(model, g):
        for region in generator_data:
            if g in generator_data[region]:
                return generator_data[region][g]['cost']
        return 0
    diag_model.c_g = pyo.Param(diag_model.G, initialize=gen_cost_init, doc="Generation cost")
    
    def gen_capacity_init(model, g):
        for region in generator_data:
            if g in generator_data[region]:
                return generator_data[region][g]['capacity']
        return 0
    diag_model.p_g_max = pyo.Param(diag_model.G, initialize=gen_capacity_init, doc="Generator capacity")
    
    def fmax_init(model, i, j):
        return fmax_data.get(i, {}).get(j, 0)
    diag_model.f_max = pyo.Param(diag_model.L, initialize=fmax_init, doc="Line capacity")

    # 변수들
    diag_model.p_g = pyo.Var(diag_model.G, bounds=lambda m, g: (0, m.p_g_max[g]))
    diag_model.f = pyo.Var(diag_model.L, bounds=lambda m, i, j: (0, m.f_max[i,j]))

    # Slack variables 추가
    diag_model.s_pos = pyo.Var(diag_model.N, bounds=(0, None), doc="Positive slack (demand shortage)")
    diag_model.s_neg = pyo.Var(diag_model.N, bounds=(0, None), doc="Negative slack (supply excess)")

    # 목적함수: slack variables의 합을 최소화
    diag_model.obj = pyo.Objective(
        expr=sum(diag_model.s_pos[i] + diag_model.s_neg[i] for i in diag_model.N),
        sense=pyo.minimize,
        doc="Minimize total slack"
    )

    # 수정된 수급 균형 제약조건
    def slack_power_balance_rule(model, i):
        total_generation = sum(model.p_g[g] for g in model.G_i[i])
        inflow = sum(model.f[j,i] for j in model.N if (j,i) in model.L)
        outflow = sum(model.f[i,j] for j in model.N if (i,j) in model.L)
        net_outflow = outflow - inflow
        return total_generation - model.d[i] - net_outflow + model.s_pos[i] - model.s_neg[i] == 0

    diag_model.slack_power_balance = pyo.Constraint(diag_model.N, rule=slack_power_balance_rule)

    print("진단용 모델 생성 완료")

    # 진단 모델 최적화
    print("\n=== 진단 모델 최적화 실행 ===")
    diag_solver = pyo.SolverFactory('gurobi')
    if not diag_solver.available():
        diag_solver = pyo.SolverFactory('glpk')

    diag_results = diag_solver.solve(diag_model, tee=False)

    if (diag_results.solver.status == SolverStatus.ok) and \
       (diag_results.solver.termination_condition == TerminationCondition.optimal):
        
        print("\n=== 진단 결과 ===")
        total_slack = pyo.value(diag_model.obj)
        print(f"총 slack: {total_slack:.2f}")
        
        if total_slack < 1e-6:
            print("모든 제약조건이 feasible합니다!")
            return True
        else:
            print("\nInfeasible 제약조건들:")
            
            for i in diag_model.N:
                s_pos_val = pyo.value(diag_model.s_pos[i])
                s_neg_val = pyo.value(diag_model.s_neg[i])
                
                if s_pos_val > 1e-6:
                    print(f"  지역 {i}: 공급 부족 {s_pos_val:.0f} MW")
                    
                if s_neg_val > 1e-6:
                    print(f"  지역 {i}: 공급 과잉 {s_neg_val:.0f} MW")
            
            return False

    else:
        print("진단 모델도 해결할 수 없습니다.")
        return False

# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("일본 전력시스템 다중발전기 LMP 분석")
    print("=" * 80)
    
    # 1단계: 모델 생성
    model = generate_lmp_model()
    
    # 2단계: 최적화 실행
    results = solve_lmp_model(model)
    
    # 3단계: 결과 확인 및 필요시 진단
    if results is not None:
        if (results.solver.status == SolverStatus.ok) and \
           (results.solver.termination_condition == TerminationCondition.optimal):
            
            print("\n✅ 최적화 성공!")
            
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print("\n⚠️  모델이 infeasible입니다. 진단을 실행합니다...")
            is_feasible = diagnose_infeasibility()
            
            if is_feasible:
                print("\n🔄  진단 결과 feasible로 나타났습니다. 재실행합니다...")
                results = solve_lmp_model(model)
            else:
                print("\n❌  실제로 infeasible입니다. 모델을 수정하세요.")
        
        else:
            print(f"\n⚠️  예상치 못한 솔버 문제:")
            print(f"     솔버 상태: {results.solver.status}")
            print(f"     종료 조건: {results.solver.termination_condition}")
    
    # 4단계: 모델 요약
    print(f"\n" + "=" * 50)
    print("모델 요약:")
    print("  - generate_lmp_model(): 모델 생성 및 정보 출력")
    print("  - solve_lmp_model(): 최적화 실행 및 결과 분석")
    print("  - diagnose_infeasibility(): infeasible 시 진단")
    print("=" * 50)