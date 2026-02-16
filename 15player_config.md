# 15-Player Energy Community Configuration

## Motivation

6인 실험에서 brute force core checking은 $2^6 - 1 = 63$ coalitions으로 tractable했으나, 15인에서는 $2^{15} - 1 = 32{,}767$ coalitions이 필요하다. 이 계산 비용이 CHP(convex hull pricing) 기반 stability 검증의 실용성을 보여주는 핵심 실험이 된다.

## 기존 6인 구성 (Baseline)

| Player | Role | Technology | Integer Variables |
|--------|------|------------|-------------------|
| u1 | Wind Producer | Wind + Electric Storage | — |
| u2 | H₂ Producer | Electrolyzer + H₂ Storage | z_on, z_sb, z_off (3-state) |
| u3 | Heat Producer | Heat Pump + Heat Storage | z_on (startup/shutdown) |
| u4 | Elec Consumer | Non-flexible elec demand | — |
| u5 | H₂ Consumer | Non-flexible H₂ demand | — |
| u6 | Heat Consumer | Non-flexible heat demand | — |

## 6인 실험에서 확인된 Failure Modes

1. **Commitment divergence** (1000hh): {u2, u5}가 grand coalition과 다른 startup schedule을 선택 → IP pricing이 alternative commitment의 가치를 반영 못함
2. **Export cap → IR violation** (export_cap_020): H₂ 잉여로 shadow price 하락 → u2 profit < 0
3. **Undersupply amplification** (1000hh): u5 demand↑ → v({u2,u5}) volume에 비례 증가, grand coalition pricing은 이를 따라가지 못함

## 신규 9인 설계

### u7: Solar + Electric Storage

```python
"players_with_solar": ["u7"]
"players_with_elec_storage": ["u1", "u7"]
```

**연구 질문**: Wind(u1)와 다른 temporal generation profile이 electricity sector의 가격 구조를 어떻게 변화시키는가? Solar peak(낮)이 electrolyzer 운영과 겹치면서 H₂ 생산 비용 구조에 영향.

**파라미터 고려사항**: Solar capacity를 wind 대비 어느 수준으로 설정할지. 너무 크면 낮 시간대 전기가격이 0으로 붕괴.

### u8: Electrolyzer (소형, 저장 없음)

```python
"players_with_electrolyzers": ["u2", "u8"]
"players_with_hydro_storage": ["u2"]  # u8은 제외
"players_with_fl_elec_demand": ["u2", "u3", "u8"]
```

**연구 질문**: H₂ 생산자 경쟁이 {u2, u5} blocking을 완화하는가? u2가 유일한 electrolyzer일 때의 독점적 outside option이 u8 추가로 희석되는지 확인.

**파라미터 고려사항**:
- `els_cap_u8 < els_cap_u2` (예: 0.5× 또는 0.7×)
- H₂ storage 없음 → u2와의 비대칭으로 pricing 차이 관찰
- Startup cost는 동일하게 유지 (c_su_G = 50)

### u9: Heat Pump (소형, 저장 없음)

```python
"players_with_heatpumps": ["u3", "u9"]
"players_with_heat_storage": ["u3"]  # u9은 제외
"players_with_fl_elec_demand": ["u2", "u3", "u8", "u9"]
```

**연구 질문**: Heat sector에도 producer 경쟁 효과가 동일하게 적용되는가? 6인에서 heat sector는 상대적으로 안정적이었는데, 이것이 u3의 독점 때문인지 수요 구조 때문인지 분리.

**파라미터 고려사항**: `hp_cap_u9 < hp_cap_u3`

### u10: Wind + Electrolyzer + H₂ Storage (Vertically Integrated)

```python
"players_with_wind": ["u1", "u10"]
"players_with_electrolyzers": ["u2", "u8", "u10"]
"players_with_hydro_storage": ["u2", "u10"]
"players_with_elec_storage": ["u1", "u7", "u10"]
```

**연구 질문**: Slides item (1) — vertically integrated player가 community에 참여할 유인이 있는가? 자체 wind로 전기 조달 가능 → grand coalition 참여 없이도 높은 standalone value. 새로운 유형의 blocking coalition 가능성.

**파라미터 고려사항**:
- Wind capacity: u1 대비 소형 (예: 0.3~0.5×)
- Electrolyzer capacity: u2 대비 소형
- 이 player의 v({u10}) 자체가 높을 수 있으므로 IR violation 가능성 주시

### u11: Pure ESS (Battery Storage Only)

```python
"players_with_elec_storage": ["u1", "u7", "u10", "u11"]
```

**연구 질문**: Slides item (2) — 생산도 소비도 없이 temporal arbitrage만으로 수익을 내는 player. IP/CHP pricing이 storage의 intertemporal value를 적절히 보상하는가? Full storage 시나리오에서 관찰된 storage pricing 왜곡이 독립 ESS player에게 더 심하게 나타나는지 확인.

**파라미터 고려사항**: Storage capacity와 power를 독립적으로 설정. 너무 크면 가격 교란, 너무 작으면 무의미.

### u12: Elec Consumer (추가)

```python
"players_with_nfl_elec_demand": ["u4", "u12"]
```

**연구 질문**: Electricity demand 증가. u4와 동일 구조, 수요량 차별화로 consumer heterogeneity 도입.

**파라미터 고려사항**: `num_households_u12`를 u4와 다르게 설정하여 비대칭 consumer.

### u13: H₂ Consumer (추가)

```python
"players_with_nfl_hydro_demand": ["u5", "u13"]
```

**연구 질문**: H₂ demand 증가 시 {u2, u5} blocking 메커니즘이 {u2, u5, u13} 또는 {u2, u13}으로 확장되는지. 1000hh 실험의 undersupply 효과를 player 수 증가로 재현.

### u14: Heat Consumer (추가)

```python
"players_with_nfl_heat_demand": ["u6", "u14"]
```

**연구 질문**: Heat sector 수요 확대. Sector별 균형 유지.

### u15: Multi-sector Consumer (Elec + H₂)

```python
"players_with_nfl_elec_demand": ["u4", "u12", "u15"]
"players_with_nfl_hydro_demand": ["u5", "u13", "u15"]
```

**연구 질문**: Cross-sector consumer가 새로운 blocking coalition 구조를 유발하는가? u15는 전기와 수소를 모두 소비하므로, {u2, u15}가 electricity + hydrogen 양 sector에서 동시에 blocking할 수 있는 가능성.

## 전체 Configuration Summary

| Player | Wind | Solar | ELZ | HP | E-Sto | H₂-Sto | H-Sto | E-dem | H₂-dem | Heat-dem | fl-E-dem | Integer |
|--------|------|-------|-----|----|-------|--------|-------|-------|--------|----------|----------|---------|
| u1  | ✓ |   |   |   | ✓ |   |   |   |   |   |   | — |
| u2  |   |   | ✓ |   |   | ✓ |   |   |   |   | ✓ | 3-state |
| u3  |   |   |   | ✓ |   |   | ✓ |   |   |   | ✓ | SU/SD |
| u4  |   |   |   |   |   |   |   | ✓ |   |   |   | — |
| u5  |   |   |   |   |   |   |   |   | ✓ |   |   | — |
| u6  |   |   |   |   |   |   |   |   |   | ✓ |   | — |
| u7  |   | ✓ |   |   | ✓ |   |   |   |   |   |   | — |
| u8  |   |   | ✓ |   |   |   |   |   |   |   | ✓ | 3-state |
| u9  |   |   |   | ✓ |   |   |   |   |   |   | ✓ | SU/SD |
| u10 | ✓ |   | ✓ |   | ✓ | ✓ |   |   |   |   | ✓ | 3-state |
| u11 |   |   |   |   | ✓ |   |   |   |   |   |   | — |
| u12 |   |   |   |   |   |   |   | ✓ |   |   |   | — |
| u13 |   |   |   |   |   |   |   |   | ✓ |   |   | — |
| u14 |   |   |   |   |   |   |   |   |   | ✓ |   | — |
| u15 |   |   |   |   |   |   |   | ✓ | ✓ |   |   | — |

**Integer variable 보유 player**: u2, u3, u8, u9, u10 (5명)

## Computational Complexity

- **Brute force core**: $2^{15} - 1 = 32{,}767$ coalition problems × MIP solve per coalition
- **CHP**: Column generation on grand coalition only, then violation check
- 이 gap이 논문의 핵심 contribution: CHP가 brute force 대비 얼마나 정확하게 core violation을 detect하는가

## 실험 계획

### Phase 1: Baseline 15인

기본 파라미터(700가구, import_factor=1.5, default storage)로 실행. CHP vs brute force 비교.

### Phase 2: Sensitivity (6인에서 발견된 regime 재현)

- Undersupply: num_households 증가 → u13 추가 효과와 compound
- Export cap: 전 player 공통 적용
- Full storage: 전 player storage 활성화

### Phase 3: Player 구성 변형

- u10 제거 (vertically integrated 없이)
- u8, u9 제거 (producer 독점 유지)
- u15 → pure elec consumer로 변경 (multi-sector 효과 분리)

## 우려 사항

1. **계산 시간**: Brute force 32K coalitions이 현실적으로 가능한지. Day 1개 기준으로 먼저 feasibility 확인 필요.
2. **파라미터 tuning**: 신규 player capacity를 잘못 설정하면 trivial한 결과 (e.g., 너무 작아서 무의미, 너무 커서 기존 player를 dominate). 6인 baseline 대비 community 총 welfare가 합리적으로 증가하는 수준으로 calibrate.
3. **u10의 복잡도**: Wind+Electrolyzer+Storage가 한 player에 몰려있으면 해당 player의 subproblem 자체가 복잡. Column generation convergence에 영향 가능.
