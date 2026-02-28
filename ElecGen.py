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


class ElectricityPriceGenerator:
    """
    Electricity Price Generator
    """
    
    def __init__(self, use_korean_price: bool = False, tou: bool = False):
        self.use_korean_price = use_korean_price
        self.tou = tou
        if not self.use_korean_price:
            self.denmark_data = self.load_and_process_denmark_data()
        else:
            self.korean_data = self.load_and_process_korean_data()
    def load_and_process_denmark_data(self):
        data = pd.read_csv('./data/case_study_2019_DK2.csv')
        # Convert date into datatime
        data["Date"] = pd.to_datetime(data["Datetime"], format="%d/%m/%Y %H:%M", errors="coerce")
        mask = data.Date.isnull()
        data.loc[mask, 'Date']= pd.to_datetime(data[mask]['Datetime'], format='%d-%m-%Y %H:%M',errors='coerce')
        # Add comumn with year, month, day, hour
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['Hour'] = data['Date'].dt.hour
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
        
        data['season'] = data['Date'].apply(get_season)
        # Set datetime column as index
        data.set_index('Date', inplace = True)
        return data
    
    def generate_price(self, import_factor: float = 1.2, month: int=1, time_horizon: int = 24):
        if not self.use_korean_price:
            data = self.denmark_data[self.denmark_data['Month'] == month].reset_index(drop=True)
            data = data[data['Day'] == 1].reset_index(drop=True)
            arr = [data[data['Hour']==t]["El_price_EUR_MWh"].values[0] for t in range(time_horizon)]
            arr = np.array(arr).astype('float64')

            if self.tou:
                import_prices = self.create_tou_import_prices(arr, month, time_horizon)
            else:
                import_prices = arr*import_factor
            return {"import": import_prices, "export": arr}
        else:
            data = self.korean_data
            data = data[data['Month'] == month].reset_index(drop=True)
            ## time_horizon이 8760이라면, 아래 isin()을 적절히 계산해야.
            plot_price_profile(data)
            
            ## use medoids ##
            medoid_profiles, medoid_days, labels, cluster_sizes = find_k_medoids_price(data, k=2)
            arr = medoid_profiles/1500*1000 ## exchange rate: kor to eur, 한국데이터는 kWh 단위이므로 MWh로 변환하려고 1000 곱해줌.
            arr = arr[0]
            
            if self.tou:
                import_prices = self.create_tou_import_prices(arr, month, time_horizon)
            else:
                import_prices = arr*import_factor
            return {"import": import_prices, "export": arr}
    def load_and_process_korean_data(self):
        """한국 전력가격 데이터 로드 및 처리"""
        
        # CSV 읽기 (cp949 또는 euc-kr 인코딩 사용)
        data = pd.read_csv('./data/kor_elec_grid_price.csv', encoding='cp949')
        
        # 컬럼명 정리 (한글 깨짐 수정)
        column_mapping = {
            data.columns[0]: 'period',  # 기간
            **{data.columns[i]: f'{i:02d}시' for i in range(1, 25)},  # 01시 ~ 24시
            data.columns[25]: 'max',     # 최대
            data.columns[26]: 'min',     # 최소
            data.columns[27]: 'weighted_avg'  # 가중평균
        }
        data = data.rename(columns=column_mapping)
        data["Date"] = pd.to_datetime(data["period"], format="%Y/%m/%d", errors="coerce")
        mask = data.Date.isnull()
        data.loc[mask, 'Date']= pd.to_datetime(data[mask]['period'], format='%Y%m/%d',errors='coerce')
        # Add comumn with year, month, day, hour
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['Hour'] = data['Date'].dt.hour
        # data.set_index('Date', inplace = True)
        return data
    def create_tou_import_prices(self, smp_prices_eur, month, time_horizon):
        """
        한국의 TOU 요금제를 기반으로 import 가격 생성
        
        한국 전력 산업용(을) 고압A 선택II 요금제 기준:
        - 경부하: 23:00~09:00 (기본요금 대비 약 60-70%)
        - 중간부하: 09:00~10:00, 12:00~13:00, 17:00~23:00 (기본요금)
        - 최대부하: 10:00~12:00, 13:00~17:00 (기본요금 대비 약 140-180%)
        """
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'
        # SMP 평균가격 계산
        season = get_season(month)
        avg_smp = np.mean(smp_prices_eur.reshape(-1, 24),axis=0)
        is_negative_price = sum(avg_smp <0) >0
        if is_negative_price:
            raise Exception("Negative price의 경우 TOU 어떻게 할지 조사 필요")
        # TOU 기본요금 = SMP 평균 + 15% 안정성 프리미엄
        tou_base = avg_smp
        
        tou_import_prices = []
        
        # 시간대별 multiplier 설정
        if season == 'summer':
            # 여름철 (7-8월): 86.9 / 146.0 / 163.8
            multipliers = {
                'off_peak': 0.595,   # 86.9/146.0
                'mid_peak': 1.000,   
                'peak': 1.122        # 163.8/146.0
            }
            print("\n[TOU 설정] 여름철 일반용(을) 고압A 선택II")
        elif season == 'winter':
            # 겨울철 (11-2월): 92.1 / 144.4 / 163.8
            multipliers = {
                'off_peak': 0.638,   # 92.1/144.4
                'mid_peak': 1.000,   
                'peak': 1.134        # 163.8/144.4
            }
            print("\n[TOU 설정] 겨울철 일반용(을) 고압A 선택II")
        else:  # spring_fall
            # 봄/가을 (3-5, 9-10월): 96.6 / 146.0
            multipliers = {
                'off_peak': 0.662,   # 96.6/146.0
                'mid_peak': 1.000,   
                'peak': 1.000        # 최대부하 없음
            }
            print("\n[TOU 설정] 봄/가을 일반용(을) 고압A 선택II")
    
        print(f"  - 경부하: {multipliers['off_peak']:.3f}x")
        print(f"  - 중간부하: {multipliers['mid_peak']:.3f}x")
        print(f"  - 최대부하: {multipliers['peak']:.3f}x")
        
        for t in range(time_horizon):
            # 계절별 시간대 구분
            if season == 'summer' or season == 'winter':
                if t >= 23 or t < 9:
                    tou_multiplier = multipliers['off_peak']
                    tou_import_prices.append(tou_base[t]*tou_multiplier)
                    period_name = "경부하"
                elif (10 <= t < 12) or (13 <= t < 17):
                    tou_multiplier = multipliers['peak']
                    tou_import_prices.append(tou_base[t]*tou_multiplier)
                    period_name = "최대부하"
                else:
                    tou_multiplier = multipliers['mid_peak']
                    tou_import_prices.append(tou_base[t]*tou_multiplier)
                    period_name = "중간부하"
            else:  # spring_fall
                if 8 <= t < 11:
                    tou_multiplier = multipliers['mid_peak']
                    tou_import_prices.append(tou_base[t]*tou_multiplier)
                    period_name = "중간부하"
                else:
                    tou_multiplier = multipliers['off_peak']
                    tou_import_prices.append(tou_base[t]*tou_multiplier)
                    period_name = "경부하"
        return np.array(tou_import_prices)

class ElectricityProdGenerator:
    """
    Electricity Production Generator
    """
    
    def __init__(self, num_units: int =1, wind_el_ratio: float = 2.0,solar_el_ratio: float = 1.0, el_cap_mw:float = 1.0):
        self.num_units = num_units
        self.wind_el_ratio = wind_el_ratio
        self.solar_el_ratio = solar_el_ratio
        self.el_cap_mw = el_cap_mw
        self.wind_cap_mw = el_cap_mw * wind_el_ratio
        self.solar_cap_mw = el_cap_mw * solar_el_ratio
        self.wind_data = self.load_and_process_wind_data()
    def load_and_process_wind_data(self):
        data = pd.read_csv('./data/case_study_2019_DK2.csv')
        # Convert date into datatime
        data["Date"] = pd.to_datetime(data["Datetime"], format="%d/%m/%Y %H:%M", errors="coerce")
        mask = data.Date.isnull()
        data.loc[mask, 'Date']= pd.to_datetime(data[mask]['Datetime'], format='%d-%m-%Y %H:%M',errors='coerce')
        # Add comumn with year, month, day, hour
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['Hour'] = data['Date'].dt.hour
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
        
        data['season'] = data['Date'].apply(get_season)
        # Set datetime column as index
        data.set_index('Date', inplace = True)
        return data
    def generate_wind_production(self, month: int = 12, time_horizon: int = 24, day: int = 1):
        if month:
            data = self.wind_data[self.wind_data['Month'] == month].reset_index(drop=True)
        else:
            data = self.wind_data
        # INSERT_YOUR_CODE

        # # 모든 day별로 wind production 값 구해서 31일치 모두 플롯
        # import matplotlib.pyplot as plt
        # unique_days = sorted(data['Day'].unique())
        # fig, axs = plt.subplots(6, 6, figsize=(20, 14))  # 6x6 grid (36칸, 31일까지 커버), 남는칸은 숨김
        # axs = axs.flatten()
        # for idx, day_val in enumerate(unique_days):
        #     daily = data[data['Day'] == day_val].reset_index(drop=True)
        #     wind_profile = daily["CP"].values / daily["CP"].max() * self.wind_cap_mw
        #     ax = axs[idx]
        #     ax.plot(wind_profile, marker='o', linestyle='-')
        #     ax.set_title(f"Day {day_val}", fontsize=9)
        #     ax.set_xticks([])
        #     ax.set_yticks([])

        # for free_idx in range(len(unique_days), len(axs)):
        #     axs[free_idx].axis('off')
        # plt.suptitle("Wind CP Profiles for All Days (Normalized)", fontsize=16)
        # plt.tight_layout(rect=[0, 0, 1, 0.97])
        # plt.savefig("wind_cp_all_days_grid.png", dpi=300)
        # plt.close()

        
        """
        wind profile 시각화하고싶으면 아래 주석 제거
        """
        # plot_wind_profile(data)
        medoid_profiles, medoid_days, labels, cluster_sizes = find_k_medoids_wind(data, k=2)
        

        # Show which actual days belong to which cluster
        print("\nCluster assignments:")
        for i in range(len(medoid_profiles)):
            cluster_days = np.where(labels == i)[0] + 1
            print(f"Cluster {i} (represented by Day {medoid_days[i]}): {cluster_days[:10]}... (total {cluster_sizes[i]} days)")


        arrs = []
        for profile in medoid_profiles:
            arr = profile*1/profile.max()*self.wind_cap_mw
            arrs.append(arr)
        arrs = np.array(arrs)

        data = data[data['Day'] == day].reset_index(drop=True)
        arrs = [data["CP"].values/data["CP"].max() * self.wind_cap_mw for _ in range(2)]
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(data["CP"].values/data["CP"].max()*self.wind_cap_mw, marker='o', linestyle='-', color='b')
        plt.xlabel("Time Period (hour)")
        plt.ylabel("Wind Production (MWh)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("wind_cp_profile.png", dpi=300)
        plt.close()

        
        return arrs
    def generate_solar_production(self, month: int=1, time_horizon: int = 24):
        """
        이건 추후에 data 수집할 예정. 지금은 단순한 임의 formula로 생성.
        """
        arr = []
        for t in range(time_horizon):
            if 6 <= t <= 18:
                solar_factor = np.exp(-((t - 12) / 3.5)**2)
                arr.append(solar_factor * self.solar_el_ratio)  # MW
            else:
                arr.append(0)
        return arr

class ElectricityLoadGenerator:
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
def plot_wind_profile(data):
    import matplotlib.pyplot as plt
    import numpy as np

    # Group data by day and plot all days on same axes
    unique_days = sorted(data['Day'].unique())
    n_days = len(unique_days)
    
    plt.figure(figsize=(12, 6))
    
    # Use colormap to differentiate days
    colors = plt.cm.viridis(np.linspace(0, 1, n_days))
    
    for idx, day in enumerate(unique_days):
        day_data = data[data['Day'] == day].sort_values('Hour')
        plt.plot(day_data['Hour'], day_data['CP'], marker='o', markersize=3, 
                linewidth=1.5, alpha=0.6, c=colors[idx], label=f'Day {day}')
    
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('CP', fontsize=12)
    plt.title(f'Wind CP by Hour - All Days Overlaid (Month {month})', fontsize=14)
    plt.xlim(-0.5, 23.5)
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig('wind.png', dpi=300, bbox_inches='tight')
    plt.close()
    return

def plot_price_profile(data):
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract 24-hour price columns
    price_col = [f"{i:02d}시" for i in range(1,25)]
    
    # Get unique days
    unique_days = sorted(data['Day'].unique())
    n_days = len(unique_days)
    
    plt.figure(figsize=(12, 6))
    
    # Use colormap to differentiate days
    colors = plt.cm.viridis(np.linspace(0, 1, n_days))
    
    hours = range(1, 25)  # 1시 to 24시
    
    for idx, day in enumerate(unique_days):
        day_data = data[data['Day'] == day]
        if len(day_data) > 0:
            prices = day_data[price_col].values[0]
            plt.plot(hours, prices, marker='o', markersize=3, 
                    linewidth=1.5, alpha=0.6, c=colors[idx], label=f'Day {day}')
    
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    month = data['Month'].iloc[0] if 'Month' in data.columns else ''
    plt.title(f'Price by Hour - All Days Overlaid (Month {month})', fontsize=14)
    plt.xlim(0.5, 24.5)
    plt.xticks(range(1, 25, 2))
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig('price.png', dpi=300, bbox_inches='tight')
    plt.close()
    return

def find_k_medoids_wind(data, k=2, max_iter=100):
    """
    Find k representative days using k-medoids clustering
    
    Parameters:
    - k: number of representative days to select
    - max_iter: maximum iterations for clustering
    
    Returns:
    - medoid_profiles: array of shape (k, 24) with k representative profiles
    - medoid_days: list of day numbers
    - labels: cluster assignment for each day
    """
    # Group by day and get 24-hour profiles
    days = data.groupby('Day')['CP'].apply(list).values
    days_matrix = np.array([day for day in days if len(day) == 24])
    n_days = len(days_matrix)
    
    # Calculate pairwise distances once
    distances = np.zeros((n_days, n_days))
    for i in range(n_days):
        for j in range(i+1, n_days):
            dist = np.sqrt(np.sum((days_matrix[i] - days_matrix[j])**2))
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Initialize medoids randomly
    np.random.seed(42)  # for reproducibility
    medoid_indices = np.random.choice(n_days, k, replace=False)
    
    # K-medoids algorithm (PAM - Partitioning Around Medoids)
    for iteration in range(max_iter):
        # Assign each day to nearest medoid
        labels = np.argmin(distances[:, medoid_indices], axis=1)
        
        # Update medoids
        new_medoid_indices = []
        for cluster_id in range(k):
            cluster_members = np.where(labels == cluster_id)[0]
            if len(cluster_members) == 0:
                # Keep old medoid if cluster is empty
                new_medoid_indices.append(medoid_indices[cluster_id])
                continue
            
            # Find point in cluster with minimum sum of distances to other points in cluster
            cluster_distances = distances[np.ix_(cluster_members, cluster_members)]
            within_cluster_costs = cluster_distances.sum(axis=1)
            best_idx = cluster_members[np.argmin(within_cluster_costs)]
            new_medoid_indices.append(best_idx)
        
        new_medoid_indices = np.array(new_medoid_indices)
        
        # Check convergence
        if np.array_equal(medoid_indices, new_medoid_indices):
            break
            
        medoid_indices = new_medoid_indices
    
    medoid_profiles = days_matrix[medoid_indices]
    medoid_days = medoid_indices + 1  # +1 because Day starts from 1
    
    # Calculate cluster statistics
    cluster_sizes = [np.sum(labels == i) for i in range(k)]
    
    """
    representative profiles 시각화 하고 싶으면 아래 주석 제거
    """

    # Visualize the two representative days
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (profile, day_num, cluster_size) in enumerate(zip(medoid_profiles, medoid_days, cluster_sizes)):
        axes[i].plot(range(24), profile, 'o-', linewidth=2, markersize=6)
        axes[i].set_xlabel('Hour')
        axes[i].set_ylabel('Capacity Factor')
        axes[i].set_title(f'Representative Day {day_num}\n({cluster_size} similar days)')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim([0, 1])
        
        # Add statistics
        axes[i].text(0.02, 0.98, f'Mean: {profile.mean():.3f}\nStd: {profile.std():.3f}\nMax: {profile.max():.3f}',
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'representative_days_k_{k}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return medoid_profiles, medoid_days, labels, cluster_sizes

def find_k_medoids_price(data, k=2, max_iter=100):
    """
    Find k representative days using k-medoids clustering for price data
    
    Parameters:
    - data: DataFrame with columns "01시" to "24시" and "Day" column
    - k: number of representative days to select
    - max_iter: maximum iterations for clustering
    
    Returns:
    - medoid_profiles: array of shape (k, 24) with k representative price profiles
    - medoid_days: list of day numbers
    - labels: cluster assignment for each day
    - cluster_sizes: list of cluster sizes
    """
    # Extract 24-hour price columns
    price_col = [f"{i:02d}시" for i in range(1,25)]
    
    # Get 24-hour profiles for each day (each row is a day)
    days_matrix = data[price_col].values
    # Filter out rows that don't have exactly 24 valid values
    valid_rows = []
    valid_day_indices = []
    for idx, row in enumerate(days_matrix):
        if len(row) == 24 and not np.isnan(row).any():
            valid_rows.append(row)
            valid_day_indices.append(idx)
    days_matrix = np.array(valid_rows)
    n_days = len(days_matrix)
    
    # Calculate pairwise distances once
    distances = np.zeros((n_days, n_days))
    for i in range(n_days):
        for j in range(i+1, n_days):
            dist = np.sqrt(np.sum((days_matrix[i] - days_matrix[j])**2))
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Initialize medoids randomly
    np.random.seed(42)  # for reproducibility
    medoid_indices = np.random.choice(n_days, k, replace=False)
    
    # K-medoids algorithm (PAM - Partitioning Around Medoids)
    for iteration in range(max_iter):
        # Assign each day to nearest medoid
        labels = np.argmin(distances[:, medoid_indices], axis=1)
        
        # Update medoids
        new_medoid_indices = []
        for cluster_id in range(k):
            cluster_members = np.where(labels == cluster_id)[0]
            if len(cluster_members) == 0:
                # Keep old medoid if cluster is empty
                new_medoid_indices.append(medoid_indices[cluster_id])
                continue
            
            # Find point in cluster with minimum sum of distances to other points in cluster
            cluster_distances = distances[np.ix_(cluster_members, cluster_members)]
            within_cluster_costs = cluster_distances.sum(axis=1)
            best_idx = cluster_members[np.argmin(within_cluster_costs)]
            new_medoid_indices.append(best_idx)
        
        new_medoid_indices = np.array(new_medoid_indices)
        
        # Check convergence
        if np.array_equal(medoid_indices, new_medoid_indices):
            break
            
        medoid_indices = new_medoid_indices
    
    medoid_profiles = days_matrix[medoid_indices]
    # Get actual day numbers from the data using valid_day_indices
    medoid_days = [data.iloc[valid_day_indices[idx]]['Day'] for idx in medoid_indices]
    
    # Calculate cluster statistics
    cluster_sizes = [np.sum(labels == i) for i in range(k)]
    
    # visualize k representative days
    # fig, axes = plt.subplots(1, k, figsize=(7*k, 5))
    # if k == 1:
    #     axes = [axes]
    
    # for i, (profile, day_num, cluster_size) in enumerate(zip(medoid_profiles, medoid_days, cluster_sizes)):
    #     axes[i].plot(range(24), profile, 'o-', linewidth=2, markersize=6)
    #     axes[i].set_xlabel('Hour')
    #     axes[i].set_ylabel('Price')
    #     axes[i].set_title(f'Representative Day {day_num}\n({cluster_size} similar days)')
    #     axes[i].grid(True, alpha=0.3)
        
    #     # Add statistics
    #     axes[i].text(0.02, 0.98, f'Mean: {profile.mean():.3f}\nStd: {profile.std():.3f}\nMax: {profile.max():.3f}',
    #                 transform=axes[i].transAxes, verticalalignment='top',
    #                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # plt.tight_layout()
    # plt.savefig(f'representative_days_price_k_{k}.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    return medoid_profiles, medoid_days, labels, cluster_sizes

