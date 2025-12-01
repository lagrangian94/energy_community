import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import pandas as pd
"""
Korean Electricity Load Generator

논문 기반 전기 수요 프로파일 생성 클래스
출처: ...

jeju_hourly_elec_load.csv: 제주도 시간대별 전기 수요 데이터 (한국전력거래소_시간별 제주전력수요 https://www.data.go.kr/data/15065239/fileData.do )
제주도 평균 1가구 전력소비량: 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Literal
from scipy import stats

class KoreanElectricityLoadGenerator:
    """
    Time-series based load profile generator
    """
    
    def __init__(self, num_households: int = 100, csv_path: str = './data/jeju_hourly_elec_load.csv'):
        self.csv_path = csv_path
        self.df = None
        self.seasonal_data = {}
        self.num_households = num_households
        self._load_and_process()

    def _load_and_process(self):
        """CSV 로드 및 계절별 데이터 처리"""
        
        self.df = pd.read_csv(self.csv_path, encoding='cp949')
        self.df['date'] = pd.to_datetime(self.df.iloc[:, 0])
        
        hour_cols = [f'{h}시' for h in range(1, 25)]
        # index와 header(첫번째 열)는 제외하고 실제 data들에 0.001을 곱함 (MWh로 변환)
        for col in hour_cols:
            # object dtype이면서 '1,005.07'처럼 천단위 구분 쉼표가 있을 수 있으므로 쉼표를 삭제 후 변환
            self.df[col] = self.df[col].astype(str).str.replace(',', '', regex=False).astype(float) * 0.001
        def get_season(date):
            month = date.month
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'
        
        self.df['season'] = self.df['date'].apply(get_season)
        
        # 계절별 처리
        for season in ['spring', 'summer', 'fall', 'winter']:
            season_df = self.df[self.df['season'] == season]
            
            if len(season_df) == 0:
                continue
            
            # 원본 데이터 (num_days, 24)
            all_profiles = season_df[hour_cols].values
            
            # 평균 패턴 (deterministic component)
            mean_pattern = all_profiles.mean(axis=0)
            
            # 잔차 계산 (stochastic component)
            residuals = all_profiles - mean_pattern  # (num_days, 24)
            
            # 잔차 통계
            residual_std = residuals.std(axis=0)  # 시간대별 잔차 표준편차
            residual_cov = np.cov(residuals.T)  # 시간대 간 공분산 행렬
            
            # 각 날의 총 변동성 (RMSE)
            daily_rmse = np.sqrt((residuals**2).mean(axis=1))
            
            # 정규화
            normalized_mean = mean_pattern / mean_pattern.mean()
            normalized_std = residual_std / mean_pattern.mean()
            
            self.seasonal_data[season] = {
                'raw_profiles': all_profiles,
                'mean_pattern': mean_pattern,
                'residuals': residuals,
                'residual_std': residual_std,
                'residual_cov': residual_cov,
                'daily_rmse': daily_rmse,
                'normalized_mean': normalized_mean,
                'normalized_std': normalized_std,
                'num_days': len(season_df)
            }
            
            print(f"✓ {season:8s}: {len(season_df):3d} days | "
                  f"Mean: {mean_pattern.mean():.0f} MWh | "
                  f"RMSE range: {daily_rmse.min():.1f}-{daily_rmse.max():.1f} MWh")

    def generate_community_load(
        self,
        monthly_base_load_mwh_per_household: float,
        season: str = 'winter',
        num_days: int = 60,
        variability: Literal['flat', 'normal', 'dynamic'] = 'normal',
        method: Literal['empirical', 'gaussian', 'bootstrap'] = 'empirical'
    ) -> np.ndarray:
        """
        커뮤니티 스케일 부하 프로파일 생성
        
        Args:
            base_load_mwh: 커뮤니티 평균 부하
            season: 계절
            num_days: 생성할 일수
            variability: 'flat', 'normal', 'dynamic'
            method: 
                - 'empirical': 실제 잔차에서 샘플링
                - 'gaussian': 정규분포 가정 (multivariate normal)
                - 'bootstrap': block bootstrap
        
        Returns:
            시간대별 부하 (kW)
        """
        num_households = self.num_households
        if season not in self.seasonal_data:
            raise ValueError(f"Season '{season}' not available")
        
        data = self.seasonal_data[season]
        
        # 평균 패턴 (정규화)
        mean_pattern = data['normalized_mean']
        
        # Variability에 따라 잔차 스케일링
        scale_factor = self._get_variability_scale(season, variability)
        
        # 방법에 따라 잔차 생성
        if method == 'empirical':
            residuals = self._sample_empirical_residuals(data, num_days, scale_factor)
        elif method == 'gaussian':
            residuals = self._sample_gaussian_residuals(data, num_days, scale_factor)
        else:  # bootstrap
            residuals = self._sample_bootstrap_residuals(data, num_days, scale_factor)
        
        # 프로파일 = 평균 + 잔차
        profiles = mean_pattern + residuals
        base_load_mwh = num_households * monthly_base_load_mwh_per_household / (30*24) # 월 평균을 시간 평균으로 변환
        # 커뮤니티 스케일 적용
        community_load = (profiles * base_load_mwh).flatten()
        
        # 음수 방지
        community_load = np.maximum(community_load, 0)
        
        return community_load

    def _get_variability_scale(self, season: str, variability: str) -> float:
        """
        Variability 옵션에 따른 스케일 팩터
        
        실제 데이터의 RMSE 분포 기반
        """
        data = self.seasonal_data[season]
        daily_rmse = data['daily_rmse']
        
        if variability == 'flat':
            # 하위 25% percentile의 평균 RMSE 사용
            target_rmse = np.percentile(daily_rmse, 25)
        elif variability == 'dynamic':
            # 상위 25% percentile의 평균 RMSE 사용
            target_rmse = np.percentile(daily_rmse, 75)
        else:  # normal
            # 중간값 사용
            target_rmse = np.median(daily_rmse)
        
        # 원본 잔차의 평균 RMSE 대비 스케일
        original_rmse = daily_rmse.mean()
        scale = target_rmse / original_rmse
        
        return scale

    def _sample_empirical_residuals(
        self, 
        data: dict, 
        num_days: int, 
        scale: float
    ) -> np.ndarray:
        """
        실제 잔차에서 랜덤 샘플링
        
        가장 간단하고 안전한 방법
        """
        residuals = data['residuals']
        num_available = len(residuals)
        np.random.seed(42)
        # 랜덤하게 날짜 선택
        if num_days <= num_available:
            indices = np.random.choice(num_available, num_days, replace=False)
        else:
            indices = np.random.choice(num_available, num_days, replace=True)
        
        sampled = residuals[indices]
        
        # 정규화된 스케일 적용
        mean_pattern = data['mean_pattern'].mean()
        normalized_residuals = (sampled / mean_pattern) * scale
        
        return normalized_residuals

    def _sample_gaussian_residuals(
        self, 
        data: dict, 
        num_days: int, 
        scale: float
    ) -> np.ndarray:
        """
        Multivariate Gaussian 가정
        
        시간대 간 상관관계 보존
        """
        mean = np.zeros(24)
        cov = data['residual_cov']
        
        # 공분산 행렬을 정규화된 스케일로 조정
        mean_pattern = data['mean_pattern'].mean()
        normalized_cov = (cov / mean_pattern**2) * (scale**2)
        
        # Multivariate normal 샘플링
        np.random.seed(42)
        sampled = np.random.multivariate_normal(mean, normalized_cov, size=num_days)
        
        return sampled

    def _sample_bootstrap_residuals(
        self, 
        data: dict, 
        num_days: int, 
        scale: float
    ) -> np.ndarray:
        """
        Block bootstrap
        
        연속된 날들의 패턴 보존 (시간적 상관성)
        """
        residuals = data['residuals']
        num_available = len(residuals)
        
        # Block size (보통 3-7일)
        block_size = min(3, num_days)
        np.random.seed(42)
        sampled = []
        remaining = num_days
        
        while remaining > 0:
            # 랜덤 시작점
            start_idx = np.random.randint(0, num_available - block_size + 1)
            block = residuals[start_idx:start_idx + min(block_size, remaining)]
            sampled.append(block)
            remaining -= len(block)
        
        sampled = np.vstack(sampled)[:num_days]
        
        # 정규화 및 스케일 적용
        mean_pattern = data['mean_pattern'].mean()
        normalized_residuals = (sampled / mean_pattern) * scale
        
        return normalized_residuals

    def compare_methods(
        self,
        monthly_base_load_mwh_per_household: float = 0.036*3,
        season: str = 'summer',
        num_days: int = 7,
        variability: str = 'normal',
        save_path: Optional[str] = None
    ):
        """세 가지 방법 비교"""
        num_households = self.num_households
        base_load_mwh = num_households * monthly_base_load_mwh_per_household / (30*24) # 월 평균을 시간 평균으로 변환
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        methods = ['empirical', 'gaussian', 'bootstrap']
        colors = {'empirical': 'blue', 'gaussian': 'green', 'bootstrap': 'red'}
        
        for ax, method in zip(axes, methods):
            load = self.generate_community_load(
                base_load_mwh=base_load_mwh,
                season=season,
                num_days=num_days,
                variability=variability,
                method=method
            )
            
            hours = np.arange(len(load))
            ax.plot(hours, load, color=colors[method], linewidth=1.5, alpha=0.8)
            ax.axhline(y=base_load_mwh, color='black', linestyle='--', linewidth=1.5)
            
            for day in range(1, num_days):
                ax.axvline(x=day*24, color='gray', linestyle='--', alpha=0.3)
            
            # 통계
            daily_loads = load.reshape(num_days, 24)
            rmse = np.sqrt(((daily_loads - base_load_mwh)**2).mean())
            
            ax.set_ylabel('Load (kW)', fontsize=11)
            ax.set_title(
                f'{method.upper()} Method | RMSE: {rmse:.2f} MWh',
                fontsize=13, fontweight='bold'
            )
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Hour', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Method comparison saved: {save_path}")
        else:
            plt.show()

    def plot_residual_analysis(self, season: str = 'summer'):
        """잔차 분석 플롯"""
        
        data = self.seasonal_data[season]
        residuals = data['residuals']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 시간대별 잔차 분포
        ax = axes[0, 0]
        ax.boxplot([residuals[:, h] for h in range(24)], 
                   positions=range(1, 25), widths=0.6)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Residual (MWh)')
        ax.set_title('Hourly Residual Distribution')
        ax.grid(True, alpha=0.3)
        
        # 2. 전체 잔차 히스토그램
        ax = axes[0, 1]
        ax.hist(residuals.flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual (MWh)')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall Residual Distribution')
        ax.grid(True, alpha=0.3)
        
        # 3. 공분산 행렬
        ax = axes[1, 0]
        im = ax.imshow(data['residual_cov'], cmap='RdBu_r', aspect='auto')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Hour')
        ax.set_title('Residual Covariance Matrix')
        plt.colorbar(im, ax=ax)
        
        # 4. Daily RMSE 분포
        ax = axes[1, 1]
        daily_rmse = data['daily_rmse']
        ax.hist(daily_rmse, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.percentile(daily_rmse, 25), color='blue', 
                  linestyle='--', label='25% (flat)')
        ax.axvline(np.percentile(daily_rmse, 50), color='green', 
                  linestyle='--', label='50% (normal)')
        ax.axvline(np.percentile(daily_rmse, 75), color='red', 
                  linestyle='--', label='75% (dynamic)')
        ax.set_xlabel('Daily RMSE (MWh)')
        ax.set_ylabel('Number of Days')
        ax.set_title('Daily Variability Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

