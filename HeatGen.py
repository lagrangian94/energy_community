import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class KoreanHeatLoadGenerator:
    """
    Generate realistic heat load profiles based on Korean building data
    Reference: Kim et al. (2021), Energies 14(14), 4284
    Busan Eco-Delta Smart Village Data
    """
    
    def __init__(self, floor_area_per_household: float = 42.4, num_households: int = 1):
        """Initialize with Korean building standards"""
        self.floor_area = floor_area_per_household
        self.num_households = num_households
        
        # From Busan Smart Village paper (kWh/m²·year)
        self.heating_intensity = 31.0
        self.cooling_intensity = 19.0
        self.dhw_annual = 2839.0  # kWh/year per household
        
        # Peak loads (W/m²)
        self.peak_heating = 25.0
        self.peak_cooling = 25.0
        
        # Calculate household-level demands
        self.heating_annual = self.heating_intensity * self.floor_area
        self.cooling_annual = self.cooling_intensity * self.floor_area
        
        # Operating hours for Korea climate
        self.heating_hours = 4320  # Nov-Apr: 180 days
        self.cooling_hours = 2880  # Jun-Sep: 120 days
        
        # Average loads (kW)
        self.heating_avg = self.heating_annual / self.heating_hours
        self.cooling_avg = self.cooling_annual / self.cooling_hours
        self.dhw_avg = self.dhw_annual / 8760
        
        # Peak loads (MW for compatibility)
        self.heating_peak = self.peak_heating * self.floor_area / 1e6
        self.cooling_peak = self.peak_cooling * self.floor_area / 1e6
    
    def get_hourly_heat_load(self, hour: int, month: int = 1) -> float:
        """
        Get total heat load for specific hour and month
        
        Args:
            hour: Hour of day (0-23)
            month: Month (1-12)
            
        Returns:
            Heat load in MW
        """
        # Heating season (Nov-Apr)
        if month in [11, 12, 1, 2, 3, 4]:
            # Seasonal factor
            if month in [1, 2]:
                seasonal = 1.3
            elif month in [12, 3]:
                seasonal = 1.0
            else:
                seasonal = 0.6
            
            # Daily pattern
            if 6 <= hour <= 9:
                daily = 1.2 + 0.3 * np.exp(-((hour - 7.5) / 1.5)**2)
            elif 17 <= hour <= 23:
                daily = 1.3 + 0.4 * np.exp(-((hour - 20) / 2)**2)
            elif 0 <= hour <= 5:
                daily = 0.7
            else:
                daily = 0.8
            
            heating = min(self.heating_avg * seasonal * daily / 1000, self.heating_peak)
        else:
            heating = 0.0
        
        # Cooling season (Jun-Sep)
        if month in [6, 7, 8, 9]:
            # Seasonal factor
            seasonal = 1.3 if month in [7, 8] else 0.8
            
            # Daily pattern
            if 12 <= hour <= 18:
                daily = 1.4 + 0.5 * np.exp(-((hour - 15) / 2)**2)
            elif 19 <= hour <= 23:
                daily = 1.0
            elif 9 <= hour <= 11:
                daily = 0.8
            else:
                daily = 0.3
            
            cooling = min(self.cooling_avg * seasonal * daily / 1000, self.cooling_peak)
            raise Exception("여름엔 Heat Pump 사용안하나? 어쨌든 냉방은 우리 모델에서 안함.")
        else:
            cooling = 0.0
        
        # DHW (year-round)
        if 6 <= hour <= 9:
            dhw_factor = 1.8 + 0.8 * np.exp(-((hour - 7.5) / 1.2)**2)
        elif 19 <= hour <= 22:
            dhw_factor = 2.0 + 1.0 * np.exp(-((hour - 20.5) / 1.2)**2)
        elif 10 <= hour <= 18:
            dhw_factor = 0.5
        else:
            dhw_factor = 0.2
        
        dhw = self.dhw_avg * dhw_factor / 1000
        
        # Total (heating/cooling + DHW, never both heating and cooling)
        # Scale by number of households
        return (max(heating, cooling) + dhw) * self.num_households

    def generate_korean_heat_demand(self, hour: int, month: int = 1) -> float:
        """
        Simple function to replace existing heat demand calculation
        
        Drop-in replacement for:
            heat_demand_kw = 60 + 30 * np.cos(2 * np.pi * (t - 3) / 24)
            heat_demand = heat_demand_kw * 0.001
        
        Usage:
            parameters[f'd_H_nfl_{u}_{t}'] = generate_korean_heat_demand(t, month=1)
        """
        generator = KoreanHeatLoadGenerator()
        return generator.get_hourly_heat_load(hour, month)



# # =============================================================================
# # [1] 건물 에너지 효율 개선 시나리오 (단열 강화, ZEB 등)
# # =============================================================================

# # 시나리오 1-1: 에너지 효율 20% 개선 (난방/냉방 intensity 감소)
# generator = KoreanHeatLoadGenerator()
# generator.customize(
#     heating_intensity=31.0 * 0.8,  # 24.8 kWh/m²·year
#     cooling_intensity=19.0 * 0.8   # 15.2 kWh/m²·year
# )

# # 시나리오 1-2: 고효율 건물 (난방/냉방 intensity 40% 감소)
# generator.customize(
#     heating_intensity=31.0 * 0.6,  # 18.6 kWh/m²·year
#     cooling_intensity=19.0 * 0.6   # 11.4 kWh/m²·year
# )

# # 시나리오 1-3: Net Zero Energy Building 수준
# generator.customize(
#     heating_intensity=31.0 * 0.5,  # 15.5 kWh/m²·year
#     cooling_intensity=19.0 * 0.5,  # 9.5 kWh/m²·year
#     peak_heating=25.0 * 0.7,       # Peak load도 감소
#     peak_cooling=25.0 * 0.7
# )

# # =============================================================================
# # [2] 기후 변화 시나리오
# # =============================================================================

# # 시나리오 2-1: 온난화 (난방 감소 -15%, 냉방 증가 +30%)
# generator.customize(
#     heating_intensity=31.0 * 0.85,  # 26.35 kWh/m²·year
#     cooling_intensity=19.0 * 1.30,  # 24.7 kWh/m²·year
#     heating_hours=4320 * 0.9,       # 난방 기간 단축
#     cooling_hours=2880 * 1.2        # 냉방 기간 연장
# )

# # 시나리오 2-2: 극심한 온난화
# generator.customize(
#     heating_intensity=31.0 * 0.7,   # 21.7 kWh/m²·year
#     cooling_intensity=19.0 * 1.5,   # 28.5 kWh/m²·year
#     heating_hours=4320 * 0.75,
#     cooling_hours=2880 * 1.4
# )

# # 시나리오 2-3: 한파 심화 (난방 증가 +20%)
# generator.customize(
#     heating_intensity=31.0 * 1.2,   # 37.2 kWh/m²·year
#     peak_heating=25.0 * 1.3         # Peak도 증가
# )

# # =============================================================================
# # [3] DHW 사용 패턴 변화
# # =============================================================================

# # 시나리오 3-1: 가구원 수 증가 (DHW 수요 +30%)
# generator.customize(
#     dhw_annual=2839.0 * 1.3  # 3690.7 kWh/year
# )

# # 시나리오 3-2: 절수형 설비 도입 (DHW 수요 -25%)
# generator.customize(
#     dhw_annual=2839.0 * 0.75  # 2129.25 kWh/year
# )

# # 시나리오 3-3: 고효율 급탕 시스템 (DHW 수요 -40%)
# generator.customize(
#     dhw_annual=2839.0 * 0.6  # 1703.4 kWh/year
# )

# # =============================================================================
# # [4] 건물 규모 변화
# # =============================================================================

# # 시나리오 4-1: 소형 주택 (면적 -30%)
# generator.customize(
#     floor_area_per_household=42.4 * 0.7  # 29.68 m²
# )

# # 시나리오 4-2: 대형 주택 (면적 +50%)
# generator.customize(
#     floor_area_per_household=42.4 * 1.5  # 63.6 m²
# )

# # 시나리오 4-3: 커뮤니티 규모 확대 (가구 수 2배)
# generator.customize(
#     num_households=2
# )

# # =============================================================================
# # [5] 복합 시나리오 (Multiple parameters)
# # =============================================================================

# # 시나리오 5-1: 미래형 친환경 커뮤니티
# # (에너지 효율 개선 + 온난화 + 절수형 설비)
# generator.customize(
#     heating_intensity=31.0 * 0.7,   # 고효율 난방
#     cooling_intensity=19.0 * 1.2,   # 온난화로 냉방 증가
#     dhw_annual=2839.0 * 0.75,       # 절수형 설비
#     peak_heating=25.0 * 0.8,
#     peak_cooling=25.0 * 1.1
# )

# # 시나리오 5-2: 최악의 경우 (비효율 건물 + 극심한 기후변화)
# generator.customize(
#     heating_intensity=31.0 * 1.1,   # 노후 건물
#     cooling_intensity=19.0 * 1.6,   # 극심한 온난화
#     dhw_annual=2839.0 * 1.2,
#     peak_heating=25.0 * 1.2,
#     peak_cooling=25.0 * 1.5
# )

# # 시나리오 5-3: Passive House Standard
# generator.customize(
#     heating_intensity=15.0,         # Passive house 기준 (≤15 kWh/m²·year)
#     cooling_intensity=19.0 * 0.6,
#     dhw_annual=2839.0 * 0.8,
#     peak_heating=10.0,              # 매우 낮은 peak load
#     peak_cooling=15.0
# )

# # =============================================================================
# # [6] Heat Pump 성능 계수 변화를 반영한 시나리오
# # =============================================================================

# # 논문에서 GSHP COP=3.5, ASHP COP=3.0 사용
# # Peak load 변화로 열펌프 용량 및 운영 특성 분석

# # 시나리오 6-1: 고효율 열펌프 대응 (Peak load 평탄화)
# generator.customize(
#     peak_heating=25.0 * 0.8,
#     peak_cooling=25.0 * 0.8
# )

# # 시나리오 6-2: TES(Thermal Energy Storage) 용량 최적화를 위한 Peak 증가
# generator.customize(
#     peak_heating=25.0 * 1.3,
#     peak_cooling=25.0 * 1.3
# )

# # =============================================================================
# # [7] 계절 운영 시간 변화
# # =============================================================================

# # 시나리오 7-1: 중간기 확대 (난방/냉방 기간 단축)
# generator.customize(
#     heating_hours=4320 * 0.8,  # 3456 hours
#     cooling_hours=2880 * 0.85  # 2448 hours
# )

# # 시나리오 7-2: 사계절 극대화 (중간기 축소)
# generator.customize(
#     heating_hours=4320 * 1.15,  # 4968 hours
#     cooling_hours=2880 * 1.20   # 3456 hours
# )