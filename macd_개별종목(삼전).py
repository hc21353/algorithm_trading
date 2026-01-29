# 필요 라이브러리 임포트
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit
import itertools
from typing import Dict, Optional
import platform
# Visualization 추가 라이브러리
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

if platform.system() == 'Darwin': # 맥
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # 윈도우
    plt.rc('font', family='Malgun Gothic')
else: # 리눅스
    plt.rc('font', family='NanumBarunGothic"')

plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


# ==========================================
# [핵심] Numba 가속 Horner 계산 (정밀도 보장)
# ==========================================
@njit(fastmath=False)  # 정밀도를 위해 False 유지
def fast_horner_calc(windows: np.ndarray, z: float) -> np.ndarray:
    n_windows, window_len = windows.shape
    result = np.zeros(n_windows)
    
    for i in range(n_windows):
        val = 0.0
        for j in range(window_len):
            val = val * z + windows[i, j]
        result[i] = val
    return result

# ==========================================
# [Class 1] 완전한 Finite Horizon MACD
# ==========================================
class FiniteHorizonMACD:
    """
    Finite Horizon Common Alpha MACD (Full Finite Ver.)
    - Short/Long MA뿐만 아니라 Signal Line까지 Finite로 계산
    - 논리적 일관성 확보: 좀비 메모리(Zombie Memory) 제거
    """
    def __init__(self, short_N: int, long_N: int, signal_N: int, alpha: float):
        self.short_N = short_N
        self.long_N = long_N
        self.signal_N = signal_N
        self.alpha = alpha
        self.z = 1.0 - alpha

    def _calculate_finite_ema(self, data: np.ndarray, N: int) -> np.ndarray:
        """
        입력 데이터(data)에 대해 N기간 Finite EMA를 계산
        - data가 Price일 수도 있고, MACD Line일 수도 있음
        """
        length = len(data)
        
        # 분모 계산 (등비수열의 합)
        if self.alpha == 0:
            denominator = N
        else:
            denominator = (1 - self.z**N) / (1 - self.z)

        ema_values = np.full(length, np.nan)
        if length < N: return ema_values

        # 1. Rolling Window 생성
        windows = sliding_window_view(data, window_shape=N)
        
        # 2. Numba 가속 함수 호출
        numerators = fast_horner_calc(windows, self.z)
        
        # 3. EMA 산출
        valid_emas = numerators / denominator
        ema_values[N-1:] = valid_emas
        
        return ema_values

    def calculate(self, df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        if df.empty: return df
        prices = df[price_col].values.astype(float)

        # 1. Finite EMA (Short & Long)
        ema_short = self._calculate_finite_ema(prices, self.short_N)
        ema_long = self._calculate_finite_ema(prices, self.long_N)

        # 2. MACD Line 계산
        macd_line = ema_short - ema_long

        # 3. Signal Line도 Finite EMA로 계산
        signal_line = self._calculate_finite_ema(macd_line, self.signal_N)

        # 4. Histogram & Result
        histogram = macd_line - signal_line

        result_df = df.copy()
        result_df['fh_macd'] = macd_line
        result_df['fh_signal'] = signal_line
        result_df['fh_hist'] = histogram
        
        return result_df


# ==========================================
# [Class 2] Standard MACD 계산 클래스 (Infinite EMA)
# ==========================================
class StandardMACD:
    """
    Traditional MACD with Infinite EMA (12, 26, 9)
    - 각 EMA는 서로 다른 alpha 값 사용:
      * Short EMA: alpha = 2/(12+1) ≈ 0.1538
      * Long EMA: alpha = 2/(26+1) ≈ 0.0741
      * Signal EMA: alpha = 2/(9+1) = 0.2
    - Infinite memory (좀비 메모리 포함)
    """
    def __init__(self, short_N: int = 12, long_N: int = 26, signal_N: int = 9):
        self.short_N = short_N
        self.long_N = long_N
        self.signal_N = signal_N
        
        # 각 EMA의 alpha 계산: 2/(N+1)
        self.alpha_short = 2.0 / (short_N + 1)
        self.alpha_long = 2.0 / (long_N + 1)
        self.alpha_signal = 2.0 / (signal_N + 1)
    
    def _calculate_infinite_ema(self, data: pd.Series, alpha: float) -> pd.Series:
        """
        Traditional Infinite EMA 계산
        - Pandas의 ewm 사용 (span 방식)
        - span = (2 - alpha) / alpha
        """
        span = (2 - alpha) / alpha
        return data.ewm(span=span, adjust=False).mean()
    
    def calculate(self, df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """
        Standard MACD 계산
        
        Returns:
            DataFrame with columns: std_macd, std_signal, std_hist
        """
        if df.empty:
            return df
        
        prices = df[price_col]
        
        # 1. Short/Long EMA (각각 다른 alpha)
        ema_short = self._calculate_infinite_ema(prices, self.alpha_short)
        ema_long = self._calculate_infinite_ema(prices, self.alpha_long)
        
        # 2. MACD Line
        macd_line = ema_short - ema_long
        
        # 3. Signal Line (또 다른 alpha)
        signal_line = self._calculate_infinite_ema(macd_line, self.alpha_signal)
        
        # 4. Histogram
        histogram = macd_line - signal_line
        
        result_df = df.copy()
        result_df['std_macd'] = macd_line
        result_df['std_signal'] = signal_line
        result_df['std_hist'] = histogram
        
        return result_df


# ==============================================================================
# [Class 3] FiniteMACDOptimizer (파라미터 탐색)
# ==============================================================================
class FiniteMACDOptimizer:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.df = self._load_data(ticker, start_date, end_date)
        self.results_df = pd.DataFrame()
        self.best_params = {}
        print(f"✅ [{ticker}] 데이터 준비 완료 ({len(self.df)} rows)")

    def _load_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        print(f"📥 {ticker} 데이터 다운로드 중...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Close']].copy()

    def backtest(self, short_N: int, long_N: int, signal_N: int, alpha: float) -> Optional[Dict]:
        try:
            # Full Finite Logic 호출
            engine = FiniteHorizonMACD(short_N, long_N, signal_N, alpha)
            indic_df = engine.calculate(self.df, price_col='Close')
            hist = indic_df['fh_hist']

            valid_idx = hist.first_valid_index()
            if valid_idx is None: return None
            
            hist = hist.loc[valid_idx:]
            price_slice = self.df.loc[valid_idx:, 'Close']

            prev_hist = hist.shift(1)
            buy_signals = (prev_hist <= 0) & (hist > 0)
            sell_signals = (prev_hist >= 0) & (hist < 0)

            buy_prices = price_slice.loc[buy_signals]
            sell_prices = price_slice.loc[sell_signals]

            if sell_prices.empty or buy_prices.empty: return None
            if sell_prices.index[0] < buy_prices.index[0]:
                sell_prices = sell_prices.iloc[1:]

            min_len = min(len(buy_prices), len(sell_prices))
            if min_len < 3: return None 

            buys = buy_prices.values[:min_len]
            sells = sell_prices.values[:min_len]
            returns = (sells - buys) / buys

            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns <= 0].sum())
            snr = gross_profit / gross_loss if gross_loss != 0 else gross_profit

            return {
                'SNR': snr, 'Trades': min_len, 'Win_Rate': len(returns[returns > 0]) / min_len,
                'Gross_Profit': gross_profit, 'Gross_Loss': gross_loss,
                'Params': {'short_N': short_N, 'long_N': long_N, 'signal_N': signal_N, 'alpha': alpha}
            }
        except: return None

    def run_optimization(self):
        print("🚀 Finite MACD 정밀 최적화 시작 (Full Finite Strategy)...")
        results = []

        # === 탐색 범위 (학술적 근거 기반) ===
        alpha_range = np.arange(0.005, 0.5, 0.005)
        short_range = range(20, 70, 1)
        long_range  = range(100, 260, 1)
        
        count = 0
        total_estim = len(alpha_range) * len(short_range) * len(long_range) * 2 
        
        for alpha in alpha_range:
            for s_n in short_range:
                for l_n in long_range:
                    if l_n < s_n * 2: continue
                    if (1 - alpha) ** l_n < 0.01: continue

                    sig_opts = sorted(list(set([max(3, int(s_n * 0.25)), max(3, int(s_n * 0.4))])))

                    for sig_n in sig_opts:
                        alpha_val = round(alpha, 4)
                        res = self.backtest(s_n, l_n, sig_n, alpha_val)
                        if res:
                            row = res['Params']
                            row.update({k: v for k, v in res.items() if k != 'Params'})
                            results.append(row)
                        count += 1
                        if count % 100 == 0: print(f"탐색 중... {count} / {total_estim}", end='\r')

        self.results_df = pd.DataFrame(results)
        if not self.results_df.empty:
            best_idx = self.results_df['SNR'].idxmax()
            self.best_params = self.results_df.loc[best_idx].to_dict()
            print(f"\n✅ 최적화 완료. Best SNR: {self.best_params['SNR']:.4f}")
            print(self.best_params)

# ==============================================================================
# [Class 4] 시각화
# ==============================================================================
class FiniteStrategyVisualizer:
    """기본 대시보드"""
    
    def __init__(self, optimizer):
        self.opt = optimizer
        self.df = optimizer.df
        self.results = optimizer.results_df
        self.best = optimizer.best_params

    def plot_dashboard(self):
        """4가지 핵심 분석 차트"""
        if self.results.empty:
            print("⚠️ 최적화 결과가 없습니다.")
            return

        fig = plt.figure(figsize=(20, 14))
        plt.suptitle(f"Finite Horizon MACD 전략 분석 보고서: {self.opt.ticker}", fontsize=20, fontweight='bold')

        # 1. 민감도 분석
        ax1 = fig.add_subplot(2, 2, 1)
        sns.lineplot(data=self.results, x='alpha', y='SNR', marker='o', 
                     errorbar=None, linewidth=2, ax=ax1, color='navy')
        ax1.axvline(self.best['alpha'], color='red', linestyle='--', label=f"Optimal Alpha={self.best['alpha']}")
        ax1.set_title("1. 민감도(Alpha)와 신뢰도(SNR)의 관계", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Alpha (Common Decay Factor)")
        ax1.set_ylabel("평균 SNR")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 효율적 경계선
        ax2 = fig.add_subplot(2, 2, 2)
        sns.scatterplot(data=self.results, x='Gross_Loss', y='Gross_Profit', 
                        hue='alpha', size='Trades', sizes=(20, 200), palette='viridis', ax=ax2, alpha=0.8)
        ax2.scatter(self.best['Gross_Loss'], self.best['Gross_Profit'], 
                    color='red', marker='*', s=400, zorder=10, label='Optimal Point')
        ax2.plot([0, self.best['Gross_Loss']], [0, self.best['Gross_Profit']], 
                 'r--', alpha=0.5, label='Max SNR Slope')
        ax2.set_title("2. 효율적 경계선 (Risk vs Reward)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("총 손실 (Risk)")
        ax2.set_ylabel("총 이익 (Reward)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 유한 가중치 감쇠
        ax3 = fig.add_subplot(2, 2, 3)
        N_long = int(self.best['long_N'])
        alpha_opt = self.best['alpha']
        days = np.arange(0, N_long + 50) 
        z = 1 - alpha_opt
        weights = np.where(days < N_long, alpha_opt * (z ** days), 0)
        ax3.plot(days, weights, label=f'Alpha={alpha_opt}, N={N_long}', color='purple', linewidth=2)
        ax3.axvline(N_long, color='red', linestyle='--', label='Finite Horizon Cutoff')
        ax3.set_title(f"3. 유한 가중치 감쇠 (Finite Memory, N={N_long})", fontsize=14, fontweight='bold')
        ax3.set_xlabel("과거 경과 일수 (Lag)")
        ax3.set_ylabel("데이터 반영 비중")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 누적 자산 곡선
        ax4 = fig.add_subplot(2, 2, 4)
        p = self.best
        engine = FiniteHorizonMACD(int(p['short_N']), int(p['long_N']), int(p['signal_N']), p['alpha'])
        indic_df = engine.calculate(self.df, price_col='Close')
        hist = indic_df['fh_hist']
        
        signal = np.where((hist.shift(1) <= 0) & (hist > 0), 1, 
                 np.where((hist.shift(1) >= 0) & (hist < 0), 0, np.nan))
        position = pd.Series(signal, index=self.df.index).ffill().fillna(0)
        
        market_ret = self.df['Close'].pct_change().fillna(0)
        strategy_ret = market_ret * position.shift(1).fillna(0)
        
        equity_strategy = (1 + strategy_ret).cumprod()
        equity_benchmark = (1 + market_ret).cumprod()
        
        total_ret = (equity_strategy.iloc[-1] - 1) * 100
        
        ax4.plot(equity_strategy.index, equity_strategy, color='red', linewidth=2, label='Strategy')
        ax4.plot(equity_benchmark.index, equity_benchmark, color='gray', linestyle='--', alpha=0.5, label='Benchmark')
        ax4.set_title(f"4. 누적 자산 곡선 (Total Return: {total_ret:.1f}%)", fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class FiniteAdvancedVisualizer:
    """심층 분석"""
    
    def __init__(self, optimizer):
        self.opt = optimizer
        self.df = optimizer.df
        self.best = optimizer.best_params
        self._prepare_data()
        
    def _prepare_data(self):
        p = self.best
        sig_n = int(p.get('signal_N', p['short_N'] * 0.3))
        
        engine = FiniteHorizonMACD(int(p['short_N']), int(p['long_N']), sig_n, p['alpha'])
        result_df = engine.calculate(self.df, price_col='Close')
        
        self.hist = result_df['fh_hist']
        self.macd = result_df['fh_macd']
        self.signal_line = result_df['fh_signal']
        
        prev_hist = self.hist.shift(1)
        self.buy_sig = (prev_hist <= 0) & (self.hist > 0)
        self.sell_sig = (prev_hist >= 0) & (self.hist < 0)
        
        signal = np.where(self.buy_sig, 1, np.where(self.sell_sig, 0, np.nan))
        self.position = pd.Series(signal, index=self.df.index).ffill().fillna(0)
        
        market_ret = self.df['Close'].pct_change().fillna(0)
        self.strategy_ret = market_ret * self.position.shift(1).fillna(0)
        
        self.cum_ret = (1 + self.strategy_ret).cumprod()
        self.running_max = self.cum_ret.cummax()
        self.drawdown = (self.cum_ret / self.running_max) - 1
        self.mdd = self.drawdown.min()

    def plot_detailed_trading(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 14), sharex=True, 
                                            gridspec_kw={'height_ratios': [3, 1, 1]})
        
        ax1.plot(self.df.index, self.df['Close'], color='black', alpha=0.3, label='Price')
        ax1.scatter(self.df.index[self.buy_sig], self.df.loc[self.buy_sig, 'Close'], 
                    marker='^', color='red', s=150, zorder=5, label='Buy Signal')
        ax1.scatter(self.df.index[self.sell_sig], self.df.loc[self.sell_sig, 'Close'], 
                    marker='v', color='blue', s=150, zorder=5, label='Sell Signal')
        ax1.fill_between(self.df.index, self.df['Close'].min(), self.df['Close'].max(), 
                         where=self.position==1, color='red', alpha=0.05, label='In Position')
        ax1.set_title(f"1. 상세 매매 타점 (Finite MACD | N={int(self.best['long_N'])}, a={self.best['alpha']})", 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        colors = np.where(self.hist >= 0, 'red', 'blue')
        ax2.bar(self.hist.index, self.hist, color=colors, alpha=0.6, width=1.0)
        ax2.plot(self.hist.index, self.macd, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='MACD Line')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_title("2. Finite MACD Histogram", fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        ax3.fill_between(self.drawdown.index, self.drawdown * 100, 0, color='red', alpha=0.3)
        ax3.plot(self.drawdown.index, self.drawdown * 100, color='red', linewidth=1)
        ax3.set_title(f"3. Drawdown Chart (MDD: {self.mdd*100:.2f}%)", fontsize=12, fontweight='bold')
        ax3.set_ylabel("낙폭 (%)")
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_return_distribution(self):
        buy_indices = self.df.index[self.buy_sig]
        sell_indices = self.df.index[self.sell_sig]
        
        if len(sell_indices) > 0 and len(buy_indices) > 0:
            if sell_indices[0] < buy_indices[0]:
                sell_indices = sell_indices[1:]
        
        min_len = min(len(buy_indices), len(sell_indices))
        
        if min_len == 0:
            print("⚠️ 유효한 거래 쌍(Buy-Sell)이 없어 분포를 그릴 수 없습니다.")
            return

        buy_indices = buy_indices[:min_len]
        sell_indices = sell_indices[:min_len]
        
        buy_prices = self.df.loc[buy_indices, 'Close'].values
        sell_prices = self.df.loc[sell_indices, 'Close'].values
        trade_returns = (sell_prices - buy_prices) / buy_prices * 100
        
        plt.figure(figsize=(10, 6))
        sns.histplot(trade_returns, bins=20, kde=True, color='purple')
        plt.axvline(0, color='black', linestyle='--')
        plt.axvline(np.mean(trade_returns), color='red', label=f'Mean: {np.mean(trade_returns):.2f}%')
        
        win_rate = len(trade_returns[trade_returns > 0]) / len(trade_returns) * 100
        stats_text = (f"총 거래: {len(trade_returns)}회\n"
                      f"승률: {win_rate:.1f}%\n"
                      f"최대 이익: {trade_returns.max():.1f}%\n"
                      f"최대 손실: {trade_returns.min():.1f}%")
        
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='right', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title("거래별 수익률 분포 (Finite Model)", fontsize=14, fontweight='bold')
        plt.xlabel("수익률 (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# ==============================================================================
# [Class 5] Enhanced Visualization Suite (파라미터 히트맵, 엘보 포인트 제거)
# ==============================================================================
class EnhancedVisualization:
    """추가 시각화 (4개만 유지)"""
    
    def __init__(self, optimizer, adv_visualizer):
        self.opt = optimizer
        self.adv_viz = adv_visualizer
        self.df = optimizer.df
        self.results = optimizer.results_df
        self.best = optimizer.best_params
        
        print(f"\n{'='*70}")
        print("✅ Enhanced Visualization Suite 초기화")
        print(f"{'='*70}")
        print(f"   최적 파라미터: α={self.best['alpha']:.4f}, " +
              f"N=({int(self.best['short_N'])}, {int(self.best['long_N'])}, {int(self.best['signal_N'])})")
        print(f"   탐색 결과: {len(self.results):,}개 조합")
        print(f"   최적 SNR: {self.best['SNR']:.4f}")
        print(f"{'='*70}\n")
    
    
    def plot_weight_distribution(self):
        """[1/4] 가중치 분포 곡선"""
        print("\n📊 [1/4] 가중치 분포 곡선 생성 중...")
        
        N_long = int(self.best['long_N'])
        N_short = int(self.best['short_N'])
        alpha_opt = self.best['alpha']
        z = 1 - alpha_opt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        days = np.arange(0, N_long + 100)
        finite_weights = np.where(days < N_long, alpha_opt * (z ** days), 0)
        infinite_weights = alpha_opt * (z ** days)
        
        ax1.plot(days, finite_weights, label=f'Finite EMA (N={N_long})', 
                color='red', linewidth=2.5, linestyle='-')
        ax1.plot(days, infinite_weights, label='Infinite EMA (Traditional)', 
                color='blue', linewidth=2, linestyle='--', alpha=0.7)
        ax1.axvline(N_long, color='black', linestyle=':', linewidth=2, 
                   label=f'Finite Cutoff (N={N_long})')
        ax1.fill_between(days[days >= N_long], 0, infinite_weights[days >= N_long],
                        color='gray', alpha=0.2, label='Zombie Memory (제거됨)')
        
        ax1.set_title(f"가중치 분포 비교: Finite vs Infinite EMA\n(α={alpha_opt:.4f})", 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel("과거 경과 일수 (Days Ago)", fontsize=11)
        ax1.set_ylabel("데이터 반영 가중치 (Weight)", fontsize=11)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, N_long + 100)
        
        explanation = ("Finite EMA는 N일 이후 가중치를 0으로 완전 절단\n"
                      "→ 오래된 데이터의 '유령 효과(Zombie Memory)' 제거\n"
                      "→ 노이즈 감소 및 신호 명확화")
        ax1.text(0.95, 0.55, explanation, transform=ax1.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))
        
        days_short = np.arange(0, N_short + 50)
        days_long = np.arange(0, N_long + 50)
        short_weights = np.where(days_short < N_short, alpha_opt * (z ** days_short), 0)
        long_weights = np.where(days_long < N_long, alpha_opt * (z ** days_long), 0)
        
        ax2.plot(days_short, short_weights, label=f'Short EMA (N={N_short})', 
                color='orange', linewidth=2.5)
        ax2.plot(days_long, long_weights, label=f'Long EMA (N={N_long})', 
                color='purple', linewidth=2.5)
        ax2.axvline(N_short, color='orange', linestyle='--', alpha=0.7)
        ax2.axvline(N_long, color='purple', linestyle='--', alpha=0.7)
        ax2.fill_between(days_short[days_short < N_short], 0, short_weights[days_short < N_short],
                        color='orange', alpha=0.15, label='Short Memory Window')
        ax2.fill_between(days_long[days_long < N_long], 0, long_weights[days_long < N_long],
                        color='purple', alpha=0.15, label='Long Memory Window')
        
        ax2.set_title("Short vs Long EMA 가중치 특성", fontsize=14, fontweight='bold')
        ax2.set_xlabel("과거 경과 일수 (Days Ago)", fontsize=11)
        ax2.set_ylabel("데이터 반영 가중치 (Weight)", fontsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        speed_text = (f"Short EMA (N={N_short}):\n"
                     f"  → 빠른 반응, 단기 변동 포착\n"
                     f"  → 메모리 윈도우 짧음\n\n"
                     f"Long EMA (N={N_long}):\n"
                     f"  → 안정적 추세 파악\n"
                     f"  → 메모리 윈도우 김")
        ax2.text(0.95, 0.55, speed_text, transform=ax2.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
        
        plt.tight_layout()
        plt.show()
        print("✅ 가중치 분포 곡선 완료\n")
    
    
    def plot_standard_vs_mutant_comparison(self):
        """
        [2/4] Standard vs Mutant 비교 (개선됨)
        
        ★ 주요 개선사항:
        1. Standard MACD는 각 EMA마다 다른 alpha 사용 (12→0.1538, 26→0.0741, 9→0.2)
        2. 상단: 절대값 비교 (막대 그래프)
        3. 중단: 상대적 개선율 (%)
        4. 하단: 상세 수치 테이블
        
        ★ 개선율 계산 방식:
        - SNR, Trades, Gross Profit: (Mutant - Standard) / Standard * 100
        - Win Rate: 절대 포인트 차이 (예: 65% - 60% = +5p)
        - Gross Loss: 손실 감소율 = (Standard - Mutant) / Standard * 100
          (손실이 줄어들면 양수)
        """
        print("\n📊 [2/4] Standard vs Mutant MACD 비교 중...")
        
        # Standard MACD 계산 (각 EMA는 다른 alpha)
        print("   Standard MACD(12,26,9) 백테스트 실행 중...")
        print("   ✓ Short EMA: α = 2/(12+1) = 0.1538")
        print("   ✓ Long EMA: α = 2/(26+1) = 0.0741")
        print("   ✓ Signal EMA: α = 2/(9+1) = 0.2000")
        
        std_engine = StandardMACD(short_N=12, long_N=26, signal_N=9)
        std_result_df = std_engine.calculate(self.opt.df, price_col='Close')
        standard_hist = std_result_df['std_hist']
        
        # 백테스트
        valid_idx = standard_hist.first_valid_index()
        if valid_idx is None:
            print("⚠️ Standard MACD 백테스트 실패\n")
            return
        
        hist = standard_hist.loc[valid_idx:]
        price_slice = self.opt.df.loc[valid_idx:, 'Close']
        
        prev_hist = hist.shift(1)
        buy_signals = (prev_hist <= 0) & (hist > 0)
        sell_signals = (prev_hist >= 0) & (hist < 0)
        
        buy_prices = price_slice.loc[buy_signals]
        sell_prices = price_slice.loc[sell_signals]
        
        if sell_prices.empty or buy_prices.empty:
            print("⚠️ Standard MACD 백테스트 실패 (신호 없음)\n")
            return
        
        if sell_prices.index[0] < buy_prices.index[0]:
            sell_prices = sell_prices.iloc[1:]
        
        min_len = min(len(buy_prices), len(sell_prices))
        if min_len < 3:
            print("⚠️ Standard MACD 백테스트 실패 (거래 부족)\n")
            return
        
        buys = buy_prices.values[:min_len]
        sells = sell_prices.values[:min_len]
        returns = (sells - buys) / buys
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns <= 0].sum())
        snr = gross_profit / gross_loss if gross_loss != 0 else gross_profit
        
        standard_result = {
            'SNR': snr,
            'Trades': min_len,
            'Win_Rate': len(returns[returns > 0]) / min_len,
            'Gross_Profit': gross_profit,
            'Gross_Loss': gross_loss
        }
        
        # 비교 데이터
        metrics = ['SNR', 'Trades', 'Win_Rate', 'Gross_Profit', 'Gross_Loss']
        standard_values = [standard_result[m] for m in metrics]
        mutant_values = [self.best[m] for m in metrics]
        
        # Win Rate를 퍼센트로 변환 (상단 그래프용)
        standard_values[2] *= 100
        mutant_values[2] *= 100
        
        # Figure
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.4)
        
        # === 상단: 절대값 비교 ===
        ax1 = fig.add_subplot(gs[0, 0])
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, standard_values, width,
                       label='Standard (12,26,9)', color='steelblue', 
                       alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x_pos + width/2, mutant_values, width,
                       label=f'Mutant ({int(self.best["short_N"])},{int(self.best["long_N"])},{int(self.best["signal_N"])})', 
                       color='crimson', alpha=0.8, edgecolor='black')
        
        # 값 표시
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_title("Standard MACD vs Mutant MACD 성과 비교 (절대값)\n" +
                     "(Standard: 각 EMA마다 다른 α 적용 - Short=0.154, Long=0.074, Signal=0.2)",
                     fontsize=13, fontweight='bold')
        ax1.set_ylabel('값 (절대값)', fontsize=11)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['SNR\n(Profit Factor)', 'Trades\n(거래 횟수)', 
                            'Win Rate\n(%)', 'Gross Profit\n(총 이익)', 
                            'Gross Loss\n(총 손실)'])
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # === 중단: 개선율 (%) ===
        ax2 = fig.add_subplot(gs[1, 0])
        
        improvements = []
        for i, metric in enumerate(metrics):
            if metric == 'Win_Rate':
                # 절대 포인트 차이
                improvement = mutant_values[i] - standard_values[i]
            elif metric == 'Gross_Loss':
                # 손실 감소율 (양수 = 개선)
                improvement = (standard_values[i] - mutant_values[i]) / standard_values[i] * 100
            else:
                # 일반 증가율
                if standard_values[i] != 0:
                    improvement = (mutant_values[i] - standard_values[i]) / standard_values[i] * 100
                else:
                    improvement = 0
            improvements.append(improvement)
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax2.bar(x_pos, improvements, color=colors, alpha=0.7, edgecolor='black')
        
        # 값 표시
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            # Win Rate와 Gross Loss는 특별 처리
            if metrics[i] == 'Win_Rate':
                label = f'{height:+.1f}p'
            elif metrics[i] == 'Gross_Loss':
                label = f'{height:+.1f}%'
            else:
                label = f'{height:+.1f}%'
            
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
        
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_title("개선율 (Mutant 기준)\n" +
                     "※ Win Rate는 포인트 차이, Gross Loss는 감소율 (양수=개선)",
                     fontsize=12, fontweight='bold')
        ax2.set_ylabel('개선율 (% 또는 p)', fontsize=11)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['SNR\n(%)', 'Trades\n(%)', 'Win Rate\n(포인트)', 
                            'Gross Profit\n(%)', 'Gross Loss\n(감소율 %)'])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # === 하단: 상세 테이블 ===
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.axis('tight')
        ax3.axis('off')
        
        table_data = [
            ['지표', 'Standard (12,26,9)', 
             f'Mutant ({int(self.best["short_N"])},{int(self.best["long_N"])},{int(self.best["signal_N"])})', 
             '개선율'],
            ['SNR', f'{standard_result["SNR"]:.3f}', f'{self.best["SNR"]:.3f}', 
             f'{improvements[0]:+.1f}%'],
            ['Trades', f'{int(standard_result["Trades"])}', f'{int(self.best["Trades"])}', 
             f'{improvements[1]:+.1f}%'],
            ['Win Rate', f'{standard_result["Win_Rate"]*100:.1f}%', 
             f'{self.best["Win_Rate"]*100:.1f}%', f'{improvements[2]:+.1f}p'],
            ['Gross Profit', f'{standard_result["Gross_Profit"]:.3f}', 
             f'{self.best["Gross_Profit"]:.3f}', f'{improvements[3]:+.1f}%'],
            ['Gross Loss', f'{standard_result["Gross_Loss"]:.3f}', 
             f'{self.best["Gross_Loss"]:.3f}', f'{improvements[4]:+.1f}%']
        ]
        
        table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.2, 0.25, 0.25, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 헤더 스타일
        for i in range(4):
            table[(0, i)].set_facecolor('lightgray')
            table[(0, i)].set_text_props(weight='bold')
        
        # 개선율 색상
        for i in range(1, 6):
            imp_val = float(table_data[i][3].replace('%', '').replace('p', ''))
            if imp_val > 0:
                table[(i, 3)].set_facecolor('lightgreen')
            elif imp_val < 0:
                table[(i, 3)].set_facecolor('lightcoral')
        
        plt.tight_layout()
        plt.show()
        
        print("\n📊 비교 요약:")
        print(f"   Standard SNR: {standard_result['SNR']:.3f}")
        print(f"   Mutant SNR:   {self.best['SNR']:.3f}")
        print(f"   개선율:       {improvements[0]:+.1f}%")
        print("\n   ✓ Standard MACD는 각 EMA마다 고유한 alpha 값을 사용했습니다.")
        print("   ✓ 상단 그래프: 절대값 비교")
        print("   ✓ 중단 그래프: 상대적 개선율 (%)")
        print("   ✓ 하단 테이블: 상세 수치")
        print("✅ Standard vs Mutant 비교 완료\n")
    
    
    def plot_histogram_zoom_comparison(self):
        """[3/4] 히스토그램 확대 비교"""
        print("\n📊 [3/4] 히스토그램 확대 비교 생성 중...")
        
        print("   Standard MACD 계산 중...")
        std_engine = StandardMACD(short_N=12, long_N=26, signal_N=9)
        standard_df = std_engine.calculate(self.opt.df, price_col='Close')
        standard_hist = standard_df['std_hist']
        
        mutant_hist = self.adv_viz.hist
        
        window_size = 60
        rolling_std = self.df['Close'].rolling(window_size).std()
        high_volatility_indices = rolling_std[rolling_std > rolling_std.quantile(0.8)].index
        
        if len(high_volatility_indices) == 0:
            print("⚠️ 적절한 확대 구간을 찾을 수 없습니다.\n")
            return
        
        mid_idx = len(high_volatility_indices) // 2
        zoom_center = high_volatility_indices[mid_idx]
        zoom_range_size = 30
        zoom_start = max(0, self.df.index.get_loc(zoom_center) - zoom_range_size)
        zoom_end = min(len(self.df), self.df.index.get_loc(zoom_center) + zoom_range_size)
        zoom_range = self.df.index[zoom_start:zoom_end]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 1, 1]})
        
        ax1.plot(zoom_range, self.df.loc[zoom_range, 'Close'],
                color='black', linewidth=2, label='Price')
        
        standard_buy = (standard_hist.shift(1) <= 0) & (standard_hist > 0)
        standard_sell = (standard_hist.shift(1) >= 0) & (standard_hist < 0)
        mutant_buy = self.adv_viz.buy_sig
        mutant_sell = self.adv_viz.sell_sig
        
        standard_buy_zoom = standard_buy.loc[zoom_range]
        standard_sell_zoom = standard_sell.loc[zoom_range]
        mutant_buy_zoom = mutant_buy.loc[zoom_range]
        mutant_sell_zoom = mutant_sell.loc[zoom_range]
        
        ax1.scatter(zoom_range[standard_buy_zoom], 
                   self.df.loc[zoom_range[standard_buy_zoom], 'Close'],
                   marker='^', s=100, color='blue', alpha=0.4, 
                   edgecolors='darkblue', linewidths=1, label='Standard Buy', zorder=3)
        ax1.scatter(zoom_range[standard_sell_zoom], 
                   self.df.loc[zoom_range[standard_sell_zoom], 'Close'],
                   marker='v', s=100, color='cyan', alpha=0.4, 
                   edgecolors='darkcyan', linewidths=1, label='Standard Sell', zorder=3)
        ax1.scatter(zoom_range[mutant_buy_zoom], 
                   self.df.loc[zoom_range[mutant_buy_zoom], 'Close'],
                   marker='^', s=200, color='red', alpha=0.9, 
                   edgecolors='darkred', linewidths=2, label='Mutant Buy', zorder=5)
        ax1.scatter(zoom_range[mutant_sell_zoom], 
                   self.df.loc[zoom_range[mutant_sell_zoom], 'Close'],
                   marker='v', s=200, color='orange', alpha=0.9, 
                   edgecolors='darkorange', linewidths=2, label='Mutant Sell', zorder=5)
        
        ax1.set_title(f"주가 차트 및 매매 신호 비교 (확대 구간)\n" +
                     f"{zoom_range[0].date()} ~ {zoom_range[-1].date()}",
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=11)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        standard_hist_zoom = standard_hist.loc[zoom_range]
        colors_standard = np.where(standard_hist_zoom >= 0, 'blue', 'cyan')
        ax2.bar(zoom_range, standard_hist_zoom, color=colors_standard, alpha=0.6, width=1.0)
        ax2.axhline(0, color='black', linewidth=1)
        
        crossing_points_std = zoom_range[(standard_buy_zoom) | (standard_sell_zoom)]
        ax2.scatter(crossing_points_std, [0] * len(crossing_points_std),
                   marker='o', s=100, color='yellow', edgecolors='black', 
                   linewidths=1.5, zorder=10)
        
        ax2.set_title("Standard MACD Histogram (12,26,9)", fontsize=12, fontweight='bold')
        ax2.set_ylabel('Histogram', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        standard_signals = pd.Series(0, index=zoom_range)
        standard_signals[standard_buy_zoom] = 1
        standard_signals[standard_sell_zoom] = -1
        
        signal_changes = standard_signals[standard_signals != 0]
        if len(signal_changes) > 1:
            intervals = np.diff(signal_changes.index.to_julian_date())
            short_intervals = intervals[intervals < 5]
            whipsaw_count = len(short_intervals)
        else:
            whipsaw_count = 0
        
        ax2.text(0.02, 0.95, f'Whipsaw 의심 신호: {whipsaw_count}회',
                transform=ax2.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        mutant_hist_zoom = mutant_hist.loc[zoom_range]
        colors_mutant = np.where(mutant_hist_zoom >= 0, 'red', 'blue')
        ax3.bar(zoom_range, mutant_hist_zoom, color=colors_mutant, alpha=0.6, width=1.0)
        ax3.axhline(0, color='black', linewidth=1)
        
        crossing_points_mut = zoom_range[(mutant_buy_zoom) | (mutant_sell_zoom)]
        ax3.scatter(crossing_points_mut, [0] * len(crossing_points_mut),
                   marker='o', s=100, color='gold', edgecolors='red', 
                   linewidths=2, zorder=10)
        
        ax3.set_title(f"Mutant MACD Histogram ({int(self.best['short_N'])},{int(self.best['long_N'])},{int(self.best['signal_N'])})",
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Histogram', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        mutant_signals = pd.Series(0, index=zoom_range)
        mutant_signals[mutant_buy_zoom] = 1
        mutant_signals[mutant_sell_zoom] = -1
        mutant_signal_count = (mutant_signals != 0).sum()
        
        ax3.text(0.02, 0.95, f'총 신호: {mutant_signal_count}회',
                transform=ax3.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        comparison_text = (f"신호 비교:\n"
                          f"  Standard: {(standard_signals != 0).sum()}회\n"
                          f"  Mutant: {mutant_signal_count}회\n"
                          f"→ Mutant는 노이즈 필터링으로\n"
                          f"  불필요한 신호 {(standard_signals != 0).sum() - mutant_signal_count}회 제거")
        ax1.text(0.98, 0.02, comparison_text, transform=ax1.transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        print("✅ 히스토그램 확대 비교 완료\n")
    
    
    def plot_ema_crossover_dynamics(self):
        """[4/4] EMA Crossover Dynamics"""
        print("\n📊 [4/4] EMA Crossover Dynamics 생성 중...")
        
        p = self.best
        engine = FiniteHorizonMACD(int(p['short_N']), int(p['long_N']), 
                                   int(p['signal_N']), p['alpha'])
        result_df = engine.calculate(self.df, price_col='Close')
        
        prices = self.df['Close'].values.astype(float)
        ema_short = engine._calculate_finite_ema(prices, int(p['short_N']))
        ema_long = engine._calculate_finite_ema(prices, int(p['long_N']))
        
        ema_short = pd.Series(ema_short, index=self.df.index)
        ema_long = pd.Series(ema_long, index=self.df.index)
        ema_diff = ema_short - ema_long
        
        prev_diff = ema_diff.shift(1)
        golden_cross = (prev_diff <= 0) & (ema_diff > 0)
        dead_cross = (prev_diff >= 0) & (ema_diff < 0)
        
        plot_days = min(252, len(self.df))
        plot_range = self.df.index[-plot_days:]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        ax1.plot(plot_range, self.df.loc[plot_range, 'Close'],
                color='black', linewidth=1.5, alpha=0.7, label='Price')
        ax1.plot(plot_range, ema_short.loc[plot_range],
                color='orange', linewidth=2, label=f'Short EMA (N={int(p["short_N"])})')
        ax1.plot(plot_range, ema_long.loc[plot_range],
                color='purple', linewidth=2, label=f'Long EMA (N={int(p["long_N"])})')
        
        gc_in_range = golden_cross.loc[plot_range]
        ax1.scatter(plot_range[gc_in_range], 
                   self.df.loc[plot_range[gc_in_range], 'Close'],
                   marker='^', s=200, color='gold', edgecolors='red', 
                   linewidths=2, zorder=10, label='Golden Cross')
        
        dc_in_range = dead_cross.loc[plot_range]
        ax1.scatter(plot_range[dc_in_range], 
                   self.df.loc[plot_range[dc_in_range], 'Close'],
                   marker='v', s=200, color='gray', edgecolors='blue', 
                   linewidths=2, zorder=10, label='Dead Cross')
        
        ax1.set_title(f"주가 및 EMA 동역학\n" +
                     f"(Short EMA: {int(p['short_N'])}, Long EMA: {int(p['long_N'])}, α={p['alpha']:.4f})",
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price / EMA', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(plot_range, ema_diff.loc[plot_range],
                color='green', linewidth=2, label='EMA Difference (Short - Long)')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.fill_between(plot_range, 0, ema_diff.loc[plot_range],
                        where=ema_diff.loc[plot_range] >= 0, 
                        color='red', alpha=0.3, label='Short > Long (상승 추세)')
        ax2.fill_between(plot_range, 0, ema_diff.loc[plot_range],
                        where=ema_diff.loc[plot_range] < 0, 
                        color='blue', alpha=0.3, label='Short < Long (하락 추세)')
        
        ax2.scatter(plot_range[gc_in_range], [0] * gc_in_range.sum(),
                   marker='^', s=200, color='gold', edgecolors='red', 
                   linewidths=2, zorder=10)
        ax2.scatter(plot_range[dc_in_range], [0] * dc_in_range.sum(),
                   marker='v', s=200, color='gray', edgecolors='blue', 
                   linewidths=2, zorder=10)
        
        ax2.set_title("EMA Difference (MACD Line)", fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Difference', fontsize=11)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        stats_text = (f"기간 내 크로스 통계:\n"
                     f"  Golden Cross: {gc_in_range.sum()}회\n"
                     f"  Dead Cross: {dc_in_range.sum()}회\n"
                     f"  평균 EMA 차이: {ema_diff.loc[plot_range].mean():.2f}")
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        print("✅ EMA Crossover Dynamics 완료\n")
    
    
    def generate_all_visualizations(self):
        """모든 추가 시각화 생성 (4개)"""
        print("\n" + "="*70)
        print("🚀 전체 추가 시각화 생성 시작 (4개)")
        print("="*70 + "\n")
        
        tasks = [
            ("가중치 분포 곡선", self.plot_weight_distribution),
            ("Standard vs Mutant 비교", self.plot_standard_vs_mutant_comparison),
            ("히스토그램 확대 비교", self.plot_histogram_zoom_comparison),
            ("EMA Crossover Dynamics", self.plot_ema_crossover_dynamics)
        ]
        
        for task_name, task_func in tqdm(tasks, desc="시각화 생성 중"):
            print(f"\n{'='*70}")
            print(f"▶ {task_name}")
            print(f"{'='*70}")
            task_func()
        
        print("\n" + "="*70)
        print("✅ 전체 추가 시각화 생성 완료! (4개)")
        print("="*70 + "\n")


class FiniteDataExporter:
    """데이터 저장"""
    def __init__(self, optimizer, visualizer):
        self.opt = optimizer
        self.viz = visualizer
        self.best = optimizer.best_params

    def get_optimization_results(self):
        df = self.opt.results_df.copy()
        if not df.empty:
            df = df.sort_values(by='SNR', ascending=False).reset_index(drop=True)
        return df

    def get_trade_log(self):
        buy_indices = self.viz.df.index[self.viz.buy_sig]
        sell_indices = self.viz.df.index[self.viz.sell_sig]
        
        trades = []
        min_len = min(len(buy_indices), len(sell_indices))
        
        if min_len > 0 and sell_indices[0] < buy_indices[0]:
            sell_indices = sell_indices[1:]
            min_len = min(len(buy_indices), len(sell_indices))

        for i in range(min_len):
            entry = buy_indices[i]
            exit = sell_indices[i]
            p_entry = self.viz.df.loc[entry, 'Close']
            p_exit = self.viz.df.loc[exit, 'Close']
            ret = (p_exit - p_entry) / p_entry
            
            trades.append({
                'Entry Date': entry, 'Entry Price': p_entry,
                'Exit Date': exit, 'Exit Price': p_exit,
                'Return (%)': ret * 100
            })
            
        return pd.DataFrame(trades)

    def save_to_excel(self, filename="finite_macd_results.xlsx"):
        print(f"💾 엑셀 저장 시작: {filename}")
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.get_optimization_results().to_excel(writer, sheet_name='Optimization', index=False)
            self.get_trade_log().to_excel(writer, sheet_name='Trade_Log', index=False)
            pd.DataFrame([self.best]).to_excel(writer, sheet_name='Best_Params', index=False)
        print("✅ 저장 완료!")


# ==============================================================================
# [MAIN] 실행
# ==============================================================================
if __name__ == "__main__":
    optimizer = FiniteMACDOptimizer(ticker="005930.ks", start_date="2013-01-01", end_date="2025-12-31")
    optimizer.run_optimization()
    
    if not optimizer.results_df.empty:
        print("\n📊 결과 시각화 및 저장 진행 중...")
        
        dashboard = FiniteStrategyVisualizer(optimizer)
        dashboard.plot_dashboard()

        adv_viz = FiniteAdvancedVisualizer(optimizer)
        adv_viz.plot_detailed_trading()
        adv_viz.plot_return_distribution()

        enhanced_viz = EnhancedVisualization(optimizer, adv_viz)
        enhanced_viz.generate_all_visualizations()      
        
        exporter = FiniteDataExporter(optimizer, adv_viz)
        exporter.save_to_excel(f"Finite_MACD_Final_{optimizer.ticker}.xlsx")