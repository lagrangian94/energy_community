"""
Korean Hydrogen Load Generator

논문 기반 수소 수요 프로파일 생성 클래스
출처: 김민수, 전성탁, 정태영 (2023), 
      "수소 충전소의 수소 판매량 데이터 분석",
      Journal of Hydrogen and New Energy, Vol. 34, No. 3, pp. 246-255
      DOI: https://doi.org/10.7316/JHNE.2023.34.3.246
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional


class KoreanHydrogenLoadGenerator:
    """
    한국 수소 충전소 데이터 기반 수요 프로파일 생성기
    
    실제 국내 주거지역 off-site 충전소의 1년간(2021.04-2022.03) 
    판매 데이터를 기반으로 승용차와 버스의 시간별 수소 수요를 생성합니다.
    
    Attributes:
    ----------
    paper_data : dict
        논문의 원본 데이터 (Fig. 4, Fig. 2, Table 1)
    car_profile : np.ndarray
        승용차 시간별 수소 수요 프로파일 (24,)
    bus_profile : np.ndarray
        버스 시간별 수소 수요 프로파일 (24,)
    metadata : dict
        데이터 메타정보
    
    Examples:
    --------
    >>> # 기본 사용
    >>> generator = KoreanHydrogenLoadGenerator()
    >>> generator.generate_profiles()
    >>> car_prof, bus_prof = generator.get_profiles()
    
    >>> # 파라미터 조정
    >>> generator.customize(avg_fill_car=4.0)
    >>> generator.summary()
    
    >>> # 시각화
    >>> generator.plot()
    """
    
    def __init__(self, verbose: bool = False):
        """
        초기화
        
        Parameters:
        ----------
        verbose : bool, default=False
            True면 생성 과정 출력
        """
        self.verbose = verbose
        
        # 논문 원본 데이터 로드
        self._load_paper_data()
        
        # 프로파일 초기화
        self.car_profile = None
        self.bus_profile = None
        
        # 생성 이력 저장
        self.generation_params = None
        
        if self.verbose:
            print("KoreanHydrogenLoadGenerator initialized")
            print(f"Data source: {self.metadata['citation']}")
    
    def _load_paper_data(self):
        """논문의 원본 데이터 로드"""
        
        # Fig. 4(a) - 승용차 시간별 연간 방문 횟수 (회/년)
        self.hourly_car_visits_annual = {
            9: 350,   # 09시
            10: 450,  # 10시
            11: 500,  # 11시
            12: 500,  # 12시
            13: 500,  # 13시
            14: 650,  # 14시 - 오후 피크
            15: 650,  # 15시
            16: 650,  # 16시
            17: 600,  # 17시
            18: 650,  # 18시
            19: 400,  # 19시
            20: 400,  # 20시
        }
        
        # Fig. 4(b) - 버스 시간별 연간 방문 횟수 (회/년)
        self.hourly_bus_visits_annual = {
            9: 50,    # 09시
            10: 220,  # 10시 - 오전 피크 시작
            11: 240,  # 11시 - 최대 피크
            12: 200,  # 12시
            13: 90,   # 13시 - 급감
            14: 50,   # 14시
            15: 55,   # 15시
            16: 53,   # 16시
            17: 50,   # 17시
            18: 45,   # 18시
            19: 40,   # 19시
            20: 35,   # 20시
        }
        
        # Fig. 2 - 평균 충전량 (kg/회)
        self.avg_fill_car = 3.54   # 승용차
        self.avg_fill_bus = 16.93  # 버스
        
        # Table 1 - 월별 일 평균 수요 (kg/일)
        self.paper_car_daily_avg = 59.20   # 12개월 평균
        self.paper_bus_daily_avg = 41.81   # 12개월 평균
        self.paper_total_daily_avg = 100.78
        
        # 메타데이터
        self.metadata = {
            'data_period': '2021.04-2022.03',
            'station_type': 'off-site',
            'location': '국내 주거 지역',
            'operating_hours': (9, 20),
            'days_per_year': 365,
            'citation': 'Kim et al. (2023), JHNE Vol.34 No.3, pp.246-255',
            'doi': '10.7316/JHNE.2023.34.3.246',
        }
        
        # paper_data 속성으로 통합
        self.paper_data = {
            'hourly_car_visits_annual': self.hourly_car_visits_annual,
            'hourly_bus_visits_annual': self.hourly_bus_visits_annual,
            'avg_fill_car': self.avg_fill_car,
            'avg_fill_bus': self.avg_fill_bus,
            'paper_car_daily_avg': self.paper_car_daily_avg,
            'paper_bus_daily_avg': self.paper_bus_daily_avg,
            'paper_total_daily_avg': self.paper_total_daily_avg,
            **self.metadata
        }
    
    def generate_profiles(self, scale_to_paper: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        기본 프로파일 생성 (논문 데이터 그대로)
        
        Parameters:
        ----------
        scale_to_paper : bool, default=True
            True면 Table 1 평균값에 맞게 스케일 조정
            False면 Fig. 4 원본 데이터 그대로
        
        Returns:
        -------
        tuple of np.ndarray (24,), (24,)
            (car_profile, bus_profile)
        """
        
        if self.verbose:
            print("\n[Generating Profiles]")
            print(f"Scale to paper average: {scale_to_paper}")
        
        # 원본 프로파일 계산
        car_raw = self._calculate_raw_profile(
            self.hourly_car_visits_annual, 
            self.avg_fill_car
        )
        bus_raw = self._calculate_raw_profile(
            self.hourly_bus_visits_annual, 
            self.avg_fill_bus
        )
        
        car_total_raw = np.sum(car_raw)
        bus_total_raw = np.sum(bus_raw)
        
        if self.verbose:
            print(f"\nRaw profiles (from Fig. 4):")
            print(f"  Car: {car_total_raw:.2f} kg/day")
            print(f"  Bus: {bus_total_raw:.2f} kg/day")
        
        # 스케일링
        if scale_to_paper:
            scale_car = self.paper_car_daily_avg / car_total_raw
            scale_bus = self.paper_bus_daily_avg / bus_total_raw
            
            self.car_profile = car_raw * scale_car
            self.bus_profile = bus_raw * scale_bus
            
            if self.verbose:
                print(f"\nScale factors:")
                print(f"  Car: {scale_car:.4f}")
                print(f"  Bus: {scale_bus:.4f}")
                print(f"\nScaled profiles (to Table 1):")
                print(f"  Car: {np.sum(self.car_profile):.2f} kg/day (target: {self.paper_car_daily_avg:.2f})")
                print(f"  Bus: {np.sum(self.bus_profile):.2f} kg/day (target: {self.paper_bus_daily_avg:.2f})")
        else:
            self.car_profile = car_raw
            self.bus_profile = bus_raw
        
        # 생성 파라미터 저장
        self.generation_params = {
            'method': 'default',
            'scale_to_paper': scale_to_paper,
            'avg_fill_car': self.avg_fill_car,
            'avg_fill_bus': self.avg_fill_bus,
        }
        
        return self.car_profile, self.bus_profile
    
    def customize(
        self,
        hourly_car_visits: Optional[Dict[int, float]] = None,
        hourly_bus_visits: Optional[Dict[int, float]] = None,
        avg_fill_car: Optional[float] = None,
        avg_fill_bus: Optional[float] = None,
        target_car_daily: Optional[float] = None,
        target_bus_daily: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        파라미터 조정된 프로파일 생성
        
        Parameters:
        ----------
        hourly_car_visits : dict, optional
            승용차 시간별 연간 방문 횟수 {hour: visits/year}
        hourly_bus_visits : dict, optional
            버스 시간별 연간 방문 횟수 {hour: visits/year}
        avg_fill_car : float, optional
            승용차 평균 충전량 (kg/회)
        avg_fill_bus : float, optional
            버스 평균 충전량 (kg/회)
        target_car_daily : float, optional
            승용차 목표 일 평균 수요 (kg/일)
        target_bus_daily : float, optional
            버스 목표 일 평균 수요 (kg/일)
        
        Returns:
        -------
        tuple of np.ndarray (24,), (24,)
            (car_profile, bus_profile)
        
        Examples:
        --------
        >>> # 승용차 충전량 10% 증가
        >>> generator.customize(avg_fill_car=3.54 * 1.1)
        
        >>> # 특정 시간대 방문 증가
        >>> custom_visits = generator.paper_data['hourly_car_visits_annual'].copy()
        >>> custom_visits[19] = 800
        >>> generator.customize(hourly_car_visits=custom_visits)
        """
        
        # 기본값 설정
        car_visits = hourly_car_visits or self.hourly_car_visits_annual
        bus_visits = hourly_bus_visits or self.hourly_bus_visits_annual
        car_fill = avg_fill_car or self.avg_fill_car
        bus_fill = avg_fill_bus or self.avg_fill_bus
        car_target = target_car_daily or self.paper_car_daily_avg
        bus_target = target_bus_daily or self.paper_bus_daily_avg
        
        if self.verbose:
            print("\n[Customizing Profiles]")
            if avg_fill_car:
                print(f"  Car avg fill: {self.avg_fill_car:.2f} → {car_fill:.2f} kg/visit")
            if avg_fill_bus:
                print(f"  Bus avg fill: {self.avg_fill_bus:.2f} → {bus_fill:.2f} kg/visit")
            if target_car_daily:
                print(f"  Car target: {self.paper_car_daily_avg:.2f} → {car_target:.2f} kg/day")
            if target_bus_daily:
                print(f"  Bus target: {self.paper_bus_daily_avg:.2f} → {bus_target:.2f} kg/day")
        
        # 원본 프로파일 계산
        car_raw = self._calculate_raw_profile(car_visits, car_fill)
        bus_raw = self._calculate_raw_profile(bus_visits, bus_fill)
        
        # 스케일링
        car_total = np.sum(car_raw)
        bus_total = np.sum(bus_raw)
        
        self.car_profile = car_raw * (car_target / car_total) if car_total > 0 else car_raw
        self.bus_profile = bus_raw * (bus_target / bus_total) if bus_total > 0 else bus_raw
        
        # 생성 파라미터 저장
        self.generation_params = {
            'method': 'customized',
            'hourly_car_visits': car_visits if hourly_car_visits else 'default',
            'hourly_bus_visits': bus_visits if hourly_bus_visits else 'default',
            'avg_fill_car': car_fill,
            'avg_fill_bus': bus_fill,
            'target_car_daily': car_target,
            'target_bus_daily': bus_target,
        }
        
        if self.verbose:
            print(f"\nGenerated profiles:")
            print(f"  Car: {np.sum(self.car_profile):.2f} kg/day")
            print(f"  Bus: {np.sum(self.bus_profile):.2f} kg/day")
        
        return self.car_profile, self.bus_profile
    
    def _calculate_raw_profile(self, hourly_visits: Dict[int, float], avg_fill: float) -> np.ndarray:
        """
        시간별 방문 횟수와 평균 충전량으로 원본 프로파일 계산
        
        Parameters:
        ----------
        hourly_visits : dict
            시간별 연간 방문 횟수 {hour: visits/year}
        avg_fill : float
            평균 충전량 (kg/회)
        
        Returns:
        -------
        np.ndarray (24,)
            시간별 일 평균 수소 수요 (kg/day/hour)
        """
        profile = np.zeros(24)
        days_per_year = self.metadata['days_per_year']
        
        for hour, annual_visits in hourly_visits.items():
            daily_visits = annual_visits / days_per_year
            daily_demand = daily_visits * avg_fill
            profile[hour] = daily_demand
        
        return profile
    
    def get_profiles(self) -> np.ndarray:
        """
        생성된 프로파일 반환
        
        Returns:
        -------
        np.ndarray (24,)
            total_profile
        
        Raises:
        ------
        ValueError
            프로파일이 아직 생성되지 않은 경우
        """
        if self.car_profile is None or self.bus_profile is None:
            self.generate_profiles()
            # raise ValueError("Profiles not generated yet. Call generate_profiles() or customize() first.")
        car_profile = self.car_profile
        bus_profile = self.bus_profile
        total_profile = car_profile + bus_profile
        return total_profile
    
    def summary(self) -> None:
        """프로파일 요약 정보 출력"""
        
        print("="*70)
        print("Korean Hydrogen Load Generator - Summary")
        print("="*70)
        
        # 데이터 출처
        print(f"\n[Data Source]")
        print(f"  Citation: {self.metadata['citation']}")
        print(f"  Period: {self.metadata['data_period']}")
        print(f"  Location: {self.metadata['location']}")
        print(f"  Station type: {self.metadata['station_type']}")
        print(f"  Operating hours: {self.metadata['operating_hours'][0]:02d}:00 - {self.metadata['operating_hours'][1]:02d}:00")
        
        # 논문 원본 데이터
        print(f"\n[Paper Data]")
        print(f"  Car avg fill: {self.avg_fill_car} kg/visit")
        print(f"  Bus avg fill: {self.avg_fill_bus} kg/visit")
        print(f"  Car daily avg (Table 1): {self.paper_car_daily_avg} kg/day")
        print(f"  Bus daily avg (Table 1): {self.paper_bus_daily_avg} kg/day")
        
        # 생성된 프로파일
        if self.car_profile is not None and self.bus_profile is not None:
            car_total = np.sum(self.car_profile)
            bus_total = np.sum(self.bus_profile)
            total = car_total + bus_total
            
            print(f"\n[Generated Profiles]")
            print(f"  Car: {car_total:.2f} kg/day")
            print(f"  Bus: {bus_total:.2f} kg/day")
            print(f"  Total: {total:.2f} kg/day")
            
            # 시간대별 분포
            car_morning = np.sum(self.car_profile[9:13])
            car_afternoon = np.sum(self.car_profile[14:19])
            bus_morning = np.sum(self.bus_profile[9:13])
            bus_afternoon = np.sum(self.bus_profile[13:21])
            
            print(f"\n[Time Distribution]")
            print(f"  Car morning (09-12h): {car_morning:.2f} kg/day ({car_morning/car_total*100:.1f}%)")
            print(f"  Car afternoon (14-18h): {car_afternoon:.2f} kg/day ({car_afternoon/car_total*100:.1f}%)")
            print(f"  Bus morning (09-12h): {bus_morning:.2f} kg/day ({bus_morning/bus_total*100:.1f}%)")
            print(f"  Bus afternoon (13-20h): {bus_afternoon:.2f} kg/day ({bus_afternoon/bus_total*100:.1f}%)")
            
            # 피크 시간
            car_peak_hour = np.argmax(self.car_profile)
            bus_peak_hour = np.argmax(self.bus_profile)
            
            print(f"\n[Peak Hours]")
            print(f"  Car: {car_peak_hour:02d}:00 ({self.car_profile[car_peak_hour]:.2f} kg/hour)")
            print(f"  Bus: {bus_peak_hour:02d}:00 ({self.bus_profile[bus_peak_hour]:.2f} kg/hour)")
            
            # 생성 파라미터
            if self.generation_params:
                print(f"\n[Generation Parameters]")
                print(f"  Method: {self.generation_params['method']}")
                for key, value in self.generation_params.items():
                    if key != 'method':
                        print(f"  {key}: {value}")
        else:
            print(f"\n[Generated Profiles]")
            print(f"  Not generated yet. Call generate_profiles() or customize() first.")
        
        print("="*70)
    
    def plot(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        프로파일 시각화
        
        Parameters:
        ----------
        save_path : str, optional
            저장 경로 (None이면 저장 안함)
        show : bool, default=True
            plt.show() 호출 여부
        """
        if self.car_profile is None or self.bus_profile is None:
            raise ValueError("Profiles not generated yet. Call generate_profiles() or customize() first.")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        hours = range(24)
        
        # 1. 승용차
        axes[0].bar(hours, self.car_profile, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('H2 Demand (kg/day/hour)', fontsize=11)
        axes[0].set_title('(a) Car Hydrogen Demand Profile', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(-0.5, 23.5)
        axes[0].axvspan(14, 19, alpha=0.2, color='orange', label='Afternoon Peak (14-18h)')
        axes[0].legend()
        axes[0].text(0.02, 0.95, f'Daily Total: {np.sum(self.car_profile):.2f} kg/day', 
                     transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. 버스
        axes[1].bar(hours, self.bus_profile, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('H2 Demand (kg/day/hour)', fontsize=11)
        axes[1].set_title('(b) Bus Hydrogen Demand Profile', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(-0.5, 23.5)
        axes[1].axvspan(9, 13, alpha=0.2, color='orange', label='Morning Peak (9-12h)')
        axes[1].legend()
        axes[1].text(0.02, 0.95, f'Daily Total: {np.sum(self.bus_profile):.2f} kg/day', 
                     transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. 전체 (Stacked)
        axes[2].bar(hours, self.car_profile, color='steelblue', alpha=0.7, 
                    edgecolor='black', label='Car')
        axes[2].bar(hours, self.bus_profile, bottom=self.car_profile, color='coral', 
                    alpha=0.7, edgecolor='black', label='Bus')
        axes[2].set_xlabel('Hour of Day', fontsize=12)
        axes[2].set_ylabel('H2 Demand (kg/day/hour)', fontsize=11)
        axes[2].set_title('(c) Total Hydrogen Demand Profile (Car + Bus)', 
                          fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(-0.5, 23.5)
        axes[2].legend()
        total_demand = np.sum(self.car_profile) + np.sum(self.bus_profile)
        axes[2].text(0.02, 0.95, f'Daily Total: {total_demand:.2f} kg/day', 
                     transform=axes[2].transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        if show:
            plt.show()
    
    def export_to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        프로파일을 numpy 배열로 export
        
        Returns:
        -------
        tuple of np.ndarray (24,), (24,)
            (car_profile, bus_profile)
        """
        arr = self.get_profiles()
        return arr
    
    def export_to_dict(self) -> Dict[str, np.ndarray]:
        """
        프로파일을 딕셔너리로 export
        
        Returns:
        -------
        dict
            {'car': car_profile, 'bus': bus_profile, 'total': total_profile}
        """
        car, bus = self.get_profiles()
        return {
            'car': car,
            'bus': bus,
            'total': car + bus
        }
    
    def validate(self) -> Dict[str, float]:
        """
        생성된 프로파일을 논문 데이터와 비교 검증
        
        Returns:
        -------
        dict
            검증 결과 (차이 %)
        """
        if self.car_profile is None or self.bus_profile is None:
            raise ValueError("Profiles not generated yet.")
        
        car_total = np.sum(self.car_profile)
        bus_total = np.sum(self.bus_profile)
        
        car_diff = ((car_total - self.paper_car_daily_avg) / self.paper_car_daily_avg) * 100
        bus_diff = ((bus_total - self.paper_bus_daily_avg) / self.paper_bus_daily_avg) * 100
        total_diff = ((car_total + bus_total - self.paper_total_daily_avg) / self.paper_total_daily_avg) * 100
        
        validation = {
            'car_diff_pct': car_diff,
            'bus_diff_pct': bus_diff,
            'total_diff_pct': total_diff,
            'car_actual': car_total,
            'bus_actual': bus_total,
            'car_target': self.paper_car_daily_avg,
            'bus_target': self.paper_bus_daily_avg,
        }
        
        print(f"\n[Validation Results]")
        print(f"  Car: {car_total:.2f} kg/day (target: {self.paper_car_daily_avg:.2f}, diff: {car_diff:+.2f}%)")
        print(f"  Bus: {bus_total:.2f} kg/day (target: {self.paper_bus_daily_avg:.2f}, diff: {bus_diff:+.2f}%)")
        print(f"  Total: {car_total + bus_total:.2f} kg/day (target: {self.paper_total_daily_avg:.2f}, diff: {total_diff:+.2f}%)")
        
        return validation

# =============================================================================
# Sensitivity Analysis 시나리오 (customize 함수 활용)
# =============================================================================

"""
아래는 sensitivity analysis를 위한 다양한 파라미터 조합 예시입니다.
각 시나리오는 customize() 메서드를 사용하여 구현할 수 있습니다.

# ============================================================================
# [1] 평균 충전량 변화 (기술 발전 / 차량 모델 변화)
# ============================================================================

# 시나리오 1-1: 승용차 소형화 (연료탱크 -20%)
generator.customize(avg_fill_car=3.54 * 0.8)  # 2.83 kg/회

# 시나리오 1-2: 승용차 대형화 (연료탱크 +20%)
generator.customize(avg_fill_car=3.54 * 1.2)  # 4.25 kg/회

# 시나리오 1-3: 승용차 대형화 (연료탱크 +50%)
generator.customize(avg_fill_car=3.54 * 1.5)  # 5.31 kg/회

# 시나리오 1-4: 버스 소형화 (-20%)
generator.customize(avg_fill_bus=16.93 * 0.8)  # 13.54 kg/회

# 시나리오 1-5: 버스 대형화 (+20%)
generator.customize(avg_fill_bus=16.93 * 1.2)  # 20.32 kg/회

# 시나리오 1-6: 버스 대형화 (+50%)
generator.customize(avg_fill_bus=16.93 * 1.5)  # 25.40 kg/회

# 시나리오 1-7: 승용차 + 버스 동시 대형화
generator.customize(
    avg_fill_car=3.54 * 1.3,   # +30%
    avg_fill_bus=16.93 * 1.3   # +30%
)


# ============================================================================
# [2] 수요 규모 변화 (보급률 / 시장 성장)
# ============================================================================

# 시나리오 2-1: 저수요 (보급률 저조, -30%)
generator.customize(
    target_car_daily=59.20 * 0.7,   # 41.44 kg/일
    target_bus_daily=41.81 * 0.7    # 29.27 kg/일
)

# 시나리오 2-2: 저수요 (-50%)
generator.customize(
    target_car_daily=59.20 * 0.5,   # 29.60 kg/일
    target_bus_daily=41.81 * 0.5    # 20.91 kg/일
)

# 시나리오 2-3: 고수요 (보급 확대, +30%)
generator.customize(
    target_car_daily=59.20 * 1.3,   # 76.96 kg/일
    target_bus_daily=41.81 * 1.3    # 54.35 kg/일
)

# 시나리오 2-4: 고수요 (+50%)
generator.customize(
    target_car_daily=59.20 * 1.5,   # 88.80 kg/일
    target_bus_daily=41.81 * 1.5    # 62.72 kg/일
)

# 시나리오 2-5: 고수요 (+100%, 2배)
generator.customize(
    target_car_daily=59.20 * 2.0,   # 118.40 kg/일
    target_bus_daily=41.81 * 2.0    # 83.62 kg/일
)

# 시나리오 2-6: 승용차만 증가 (버스 보급은 정체)
generator.customize(
    target_car_daily=59.20 * 1.5,   # +50%
    target_bus_daily=41.81 * 1.0    # 유지
)

# 시나리오 2-7: 버스만 증가 (노선 확대)
generator.customize(
    target_car_daily=59.20 * 1.0,   # 유지
    target_bus_daily=41.81 * 2.0    # +100%
)


# ============================================================================
# [3] 시간대별 패턴 변화 (충전 행동 변화)
# ============================================================================

# 시나리오 3-1: 저녁 피크 증가 (퇴근 후 충전 선호)
custom_car_visits_evening = {
    9: 350, 10: 450, 11: 500, 12: 500, 13: 500,
    14: 650, 15: 650, 16: 650, 17: 600, 
    18: 900, 19: 900, 20: 900  # 저녁 시간대 증가
}
generator.customize(hourly_car_visits=custom_car_visits_evening)

# 시나리오 3-2: 오전 피크 증가 (출근 전 충전 선호)
custom_car_visits_morning = {
    9: 700, 10: 800, 11: 700, 12: 500,  # 오전 증가
    13: 500, 14: 400, 15: 400, 16: 400, 
    17: 400, 18: 400, 19: 300, 20: 300
}
generator.customize(hourly_car_visits=custom_car_visits_morning)

# 시나리오 3-3: 균등 분포 (시간대 무관)
custom_car_visits_uniform = {
    hour: 500 for hour in range(9, 21)  # 09-20시 균등
}
generator.customize(hourly_car_visits=custom_car_visits_uniform)

# 시나리오 3-4: 점심시간 피크 (12-14시)
custom_car_visits_lunch = {
    9: 300, 10: 400, 11: 500, 
    12: 800, 13: 800, 14: 800,  # 점심시간 피크
    15: 500, 16: 500, 17: 500, 
    18: 400, 19: 300, 20: 300
}
generator.customize(hourly_car_visits=custom_car_visits_lunch)

# 시나리오 3-5: 버스 패턴 변화 (오후 운행 증가)
custom_bus_visits_afternoon = {
    9: 50, 10: 100, 11: 100, 12: 100, 13: 90,
    14: 150, 15: 180, 16: 180, 17: 150,  # 오후 증가
    18: 120, 19: 100, 20: 80
}
generator.customize(hourly_bus_visits=custom_bus_visits_afternoon)

# 시나리오 3-6: 24시간 운영 (심야 충전 포함)
custom_car_visits_24h = {
    0: 50, 1: 30, 2: 20, 3: 20, 4: 30, 5: 50,    # 심야
    6: 100, 7: 200, 8: 300, 9: 350, 10: 450,     # 오전
    11: 500, 12: 500, 13: 500, 14: 650, 15: 650, # 낮
    16: 650, 17: 600, 18: 650, 19: 400, 20: 400, # 저녁
    21: 300, 22: 200, 23: 100                     # 밤
}
generator.customize(hourly_car_visits=custom_car_visits_24h)


# ============================================================================
# [4] 복합 시나리오 (실제 시장 상황 반영)
# ============================================================================

# 시나리오 4-1: 보급 확대 + 대형화 (시장 성장기)
generator.customize(
    avg_fill_car=3.54 * 1.2,        # 차량 대형화
    avg_fill_bus=16.93 * 1.3,       # 버스 대형화
    target_car_daily=59.20 * 1.5,   # 보급 확대
    target_bus_daily=41.81 * 1.5
)

# 시나리오 4-2: 경제 침체 + 소형화 (수요 감소기)
generator.customize(
    avg_fill_car=3.54 * 0.9,        # 차량 소형화
    avg_fill_bus=16.93 * 0.9,       
    target_car_daily=59.20 * 0.7,   # 보급 감소
    target_bus_daily=41.81 * 0.7
)

# 시나리오 4-3: 주말 패턴 (레저 충전 증가)
custom_weekend_visits = {
    9: 200, 10: 300, 11: 400, 12: 500,   # 아침 늦게 시작
    13: 600, 14: 800, 15: 800, 16: 800,  # 오후 피크
    17: 700, 18: 600, 19: 500, 20: 400   # 저녁 높음
}
generator.customize(
    hourly_car_visits=custom_weekend_visits,
    target_car_daily=59.20 * 1.2  # 주말 수요 증가
)

# 시나리오 4-4: 고속도로 휴게소 (장거리 이동)
custom_highway_visits = {
    hour: 400 for hour in range(6, 23)  # 균등 분포
}
generator.customize(
    hourly_car_visits=custom_highway_visits,
    avg_fill_car=5.0,   # 만충 (장거리)
    avg_fill_bus=22.0,  # 대형 버스
    target_car_daily=80.0,
    target_bus_daily=60.0
)

# 시나리오 4-5: 도심 상업지구 (출퇴근 집중)
custom_commercial_visits = {
    9: 800, 10: 700, 11: 600, 12: 500,   # 출근 시간 집중
    13: 400, 14: 400, 15: 400, 16: 500,  
    17: 600, 18: 800, 19: 700, 20: 600   # 퇴근 시간 집중
}
generator.customize(
    hourly_car_visits=custom_commercial_visits,
    target_car_daily=70.0
)

# 시나리오 4-6: 계절별 변동 (여름 - 에어컨 사용 증가)
generator.customize(
    avg_fill_car=3.54 * 1.15,       # 연비 악화로 충전량 증가
    target_car_daily=59.20 * 1.1    # 충전 빈도 증가
)

# 시나리오 4-7: 계절별 변동 (겨울 - 히터 사용 증가)
generator.customize(
    avg_fill_car=3.54 * 1.2,        # 연비 더 악화
    target_car_daily=59.20 * 1.15   # 충전 빈도 더 증가
)


# ============================================================================
# [5] 극단 시나리오 (Stress Testing)
# ============================================================================

# 시나리오 5-1: 최소 수요 (초기 시장)
generator.customize(
    avg_fill_car=3.54 * 0.8,
    avg_fill_bus=16.93 * 0.8,
    target_car_daily=59.20 * 0.3,   # -70%
    target_bus_daily=41.81 * 0.3
)

# 시나리오 5-2: 최대 수요 (포화 시장)
generator.customize(
    avg_fill_car=3.54 * 1.5,
    avg_fill_bus=16.93 * 1.5,
    target_car_daily=59.20 * 3.0,   # +200%
    target_bus_daily=41.81 * 3.0
)

# 시나리오 5-3: 단일 피크 (극도 집중)
custom_single_peak = {
    9: 100, 10: 200, 11: 300, 
    12: 2000,  # 12시에 집중
    13: 300, 14: 200, 15: 100, 
    16: 100, 17: 100, 18: 100, 19: 100, 20: 100
}
generator.customize(hourly_car_visits=custom_single_peak)

# 시나리오 5-4: 완전 균등 (이론적 상황)
custom_perfect_uniform = {
    hour: 500 for hour in range(24)  # 24시간 균등
}
generator.customize(hourly_car_visits=custom_perfect_uniform)


# ============================================================================
# [6] Sensitivity Analysis 실행 예시
# ============================================================================

# Example: 평균 충전량에 대한 sensitivity 분석
# avg_fill_multipliers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
# results = []
# 
# for multiplier in avg_fill_multipliers:
#     gen = KoreanHydrogenLoadGenerator()
#     car_prof, bus_prof = gen.customize(
#         avg_fill_car=3.54 * multiplier,
#         avg_fill_bus=16.93 * multiplier
#     )
#     results.append({
#         'multiplier': multiplier,
#         'car_total': car_prof.sum(),
#         'bus_total': bus_prof.sum(),
#         'total': car_prof.sum() + bus_prof.sum()
#     })
# 
# # 결과 분석
# import pandas as pd
# df = pd.DataFrame(results)
# print(df)


# Example: 수요 규모에 대한 sensitivity 분석
# demand_multipliers = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
# results = []
# 
# for multiplier in demand_multipliers:
#     gen = KoreanHydrogenLoadGenerator()
#     car_prof, bus_prof = gen.customize(
#         target_car_daily=59.20 * multiplier,
#         target_bus_daily=41.81 * multiplier
#     )
#     
#     # 피크 시간 분석
#     car_peak_hour = car_prof.argmax()
#     car_peak_value = car_prof.max()
#     
#     results.append({
#         'multiplier': multiplier,
#         'car_total': car_prof.sum(),
#         'bus_total': bus_prof.sum(),
#         'car_peak_hour': car_peak_hour,
#         'car_peak_value': car_peak_value,
#         'peak_to_avg_ratio': car_peak_value / car_prof.mean()
#     })
# 
# df = pd.DataFrame(results)
# print(df)


# Example: 시간대별 패턴에 대한 sensitivity 분석
# peak_shift_scenarios = {
#     'morning': {9: 700, 10: 800, 11: 700, 12: 500, 13: 500, 14: 400, 
#                 15: 400, 16: 400, 17: 400, 18: 400, 19: 300, 20: 300},
#     'afternoon': {9: 350, 10: 450, 11: 500, 12: 500, 13: 500, 14: 650, 
#                   15: 650, 16: 650, 17: 600, 18: 650, 19: 400, 20: 400},
#     'evening': {9: 350, 10: 450, 11: 500, 12: 500, 13: 500, 14: 650, 
#                 15: 650, 16: 650, 17: 600, 18: 900, 19: 900, 20: 900},
# }
# 
# results = {}
# for scenario_name, visits in peak_shift_scenarios.items():
#     gen = KoreanHydrogenLoadGenerator()
#     car_prof, bus_prof = gen.customize(hourly_car_visits=visits)
#     
#     results[scenario_name] = {
#         'total': car_prof.sum(),
#         'peak_hour': car_prof.argmax(),
#         'peak_value': car_prof.max(),
#         'morning_pct': car_prof[9:13].sum() / car_prof.sum() * 100,
#         'afternoon_pct': car_prof[14:19].sum() / car_prof.sum() * 100,
#         'evening_pct': car_prof[18:21].sum() / car_prof.sum() * 100,
#     }
# 
# import pandas as pd
# df = pd.DataFrame(results).T
# print(df)

"""