import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d

# ----------------------------------------------------------------
# 1. 설정 및 초기화
# ----------------------------------------------------------------

# 논문 Table 2 파라미터 적용
BETA = 10 ** 3  # 30 dB
BETA_HAT = 10 ** 4  # 40 dB

# 날씨 파라미터 (Clear Sky는 학습용, Rain/Fog는 테스트용)
# Clear Sky (Reference for Non-Adaptive)
RHO_CLEAR = 1.0
RHO_HAT_CLEAR = 1.0

# Rain Conditions
RHO_RAIN = 0.8
RHO_HAT_RAIN = 0.3

# Fog Conditions
RHO_FOG = 0.3
RHO_HAT_FOG = 0.8

# P_H 범위 (Figure 6 x축: 7~14 dBm)
PH_RANGE_DBM = np.arange(5, 20, 1)

# 변조 방식 매핑
FSO_MODS = {1: 'OOK', 2: '4-PAM', 3: '8-PAM', 4: '16-PAM'}
RF_MODS = {1: 'BPSK', 2: 'QPSK', 3: '8PSK', 4: '16PSK'}


def load_lookup_table(filename='mi_lookup_tables.pkl'):
    """pkl 파일에서 MI Lookup Table을 불러옵니다."""
    try:
        with open(filename, 'rb') as f:
            luts = pickle.load(f)
        print(f"Lookup Table '{filename}' 로드 성공.")
        return luts
    except FileNotFoundError:
        print(f"오류: '{filename}'을 찾을 수 없습니다. LUT 생성 코드를 먼저 실행해주세요.")
        return None


# ----------------------------------------------------------------
# 2. 계산 함수 (JPC 및 SNR)
# ----------------------------------------------------------------

def get_snr_values(p_h_watt, alpha, m, m_hat, rho, rho_hat, beta, beta_hat):
    """
    주어진 총 전력과 분배 비율, 날씨 조건에 따른 채널별 SNR(Linear) 계산

    Parameters:
        p_h_watt: 총 전력 (Watt)
        alpha: FSO 전력 할당 비율 (0.0 ~ 1.0), P_o = alpha * P_H
        rho, rho_hat: 현재 날씨 계수

    Returns:
        snr_fso_lin, snr_rf_lin
    """
    p_o = alpha * p_h_watt
    p_r = (1 - alpha) * p_h_watt

    # SNR 공식 유도 (논문 수식 및 코드 로직 기반)
    # P/lambda (FSO SNR) = P_o * rho * beta * (m + m_hat)
    # Es/N0 (RF SNR)     = P_r * rho_hat * beta_hat * (m + m_hat)

    # 공통 항
    mod_sum = m + m_hat

    snr_fso_lin = p_o * rho * beta * mod_sum
    snr_rf_lin = p_r * rho_hat * beta_hat * mod_sum

    return snr_fso_lin, snr_rf_lin


def get_mi(luts, snr_db, mod_idx, channel_type='FSO'):
    """Lookup Table을 이용해 MI 반환"""
    if channel_type == 'FSO':
        key = FSO_MODS.get(int(mod_idx))
    else:
        key = RF_MODS.get(int(mod_idx))

    if key is None or luts is None:
        return 0.0

    try:
        mi = luts[key](snr_db)
        # MI 상한 제한 (Theoretical max)
        max_mi = np.log2(2 ** int(mod_idx) if channel_type == 'FSO' and int(mod_idx) > 1 else (
            2 if int(mod_idx) == 1 else int(mod_idx) * 1))

        # LUT 보간 오차로 인한 범위 이탈 방지
        mi = float(mi)
        return max(0.0, min(mi, max_mi))
    except Exception:
        return 0.0


# ----------------------------------------------------------------
# 3. Non-Adaptive 시스템 구현
# ----------------------------------------------------------------

def find_fixed_configuration(luts, p_h_dbm_list):
    """
    Step 1: 'Clear Sky' 조건에서 최적의 파라미터(alpha, m, m_hat)를 찾습니다.
    이 설정값들은 날씨가 변해도 고정(Fixed)되어 사용됩니다.
    """
    fixed_configs = {}  # Key: P_H_dBm, Value: (best_alpha, best_m, best_m_hat)

    print("\n[Training Phase] Clear Sky 조건 최적화 수행 중...")

    for p_h_dbm in p_h_dbm_list:
        p_h_watt = 10 ** ((p_h_dbm - 30) / 10.0)  # dBm to Watt

        best_mi = -np.inf
        best_params = (0.5, 1, 1)  # default

        # Grid Search for Parameters
        # 1. Modulation Combinations
        for m in range(1, 5):
            for m_hat in range(1, 5):
                # 2. Power Allocation (Alpha: 0.0 to 1.0)
                # FSO에 전력을 얼마나 줄 것인가? (Coarse Grid)
                for alpha in np.linspace(0.01, 0.99, 50):

                    # Clear Sky SNR 계산
                    snr_fso_lin, snr_rf_lin = get_snr_values(
                        p_h_watt, alpha, m, m_hat,
                        RHO_CLEAR, RHO_HAT_CLEAR, BETA, BETA_HAT
                    )

                    snr_fso_db = 10 * np.log10(snr_fso_lin) if snr_fso_lin > 0 else -100
                    snr_rf_db = 10 * np.log10(snr_rf_lin) if snr_rf_lin > 0 else -100

                    mi_fso = get_mi(luts, snr_fso_db, m, 'FSO')
                    mi_rf = get_mi(luts, snr_rf_db, m_hat, 'RF')

                    total_mi = mi_fso + mi_rf

                    if total_mi > best_mi:
                        best_mi = total_mi
                        best_params = (alpha, m, m_hat)

        fixed_configs[p_h_dbm] = best_params
        # print(f"  P_H={p_h_dbm}dBm: Best Alpha={best_params[0]:.2f}, m={best_params[1]}, m_hat={best_params[2]}")

    return fixed_configs


def evaluate_non_adaptive(luts, p_h_dbm_list, fixed_configs, target_rho, target_rho_hat):
    """
    Step 2: 고정된 설정값(fixed_configs)을 사용하여
    실제 날씨(target_rho, target_rho_hat)에서의 성능을 평가합니다.
    """
    mi_results = []

    for p_h_dbm in p_h_dbm_list:
        p_h_watt = 10 ** ((p_h_dbm - 30) / 10.0)

        # 저장된 설정 불러오기 (Non-Adaptive의 핵심)
        alpha, m, m_hat = fixed_configs[p_h_dbm]

        # 변경된 날씨에서의 SNR 계산 (설정값은 그대로 유지)
        snr_fso_lin, snr_rf_lin = get_snr_values(
            p_h_watt, alpha, m, m_hat,
            target_rho, target_rho_hat, BETA, BETA_HAT
        )

        snr_fso_db = 10 * np.log10(snr_fso_lin) if snr_fso_lin > 0 else -100
        snr_rf_db = 10 * np.log10(snr_rf_lin) if snr_rf_lin > 0 else -100

        # MI Lookup
        mi_fso = get_mi(luts, snr_fso_db, m, 'FSO')
        mi_rf = get_mi(luts, snr_rf_db, m_hat, 'RF')

        mi_results.append(mi_fso + mi_rf)

    return mi_results


# ----------------------------------------------------------------
# 4. 메인 실행 및 그래프 출력
# ----------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load LUT
    luts = load_lookup_table()

    if luts:
        # 2. Train: Find Fixed Configurations (Optimization under Clear Sky)
        fixed_configs = find_fixed_configuration(luts, PH_RANGE_DBM)

        # 3. Test: Evaluate under Bad Weather
        print("\n[Testing Phase] Rain/Fog 조건 성능 평가 중...")

        # Rain Condition
        mi_rain_fixed = evaluate_non_adaptive(
            luts, PH_RANGE_DBM, fixed_configs, RHO_RAIN, RHO_HAT_RAIN
        )

        # Fog Condition
        mi_fog_fixed = evaluate_non_adaptive(
            luts, PH_RANGE_DBM, fixed_configs, RHO_FOG, RHO_HAT_FOG
        )

        # 4. Plotting (Figure 6 Style)
        plt.figure(figsize=(8, 6))

        # Rain (Non-Adaptive) - Blue Star marker
        plt.plot(PH_RANGE_DBM, mi_rain_fixed, 'b*:', markersize=10, linewidth=1.5, label='Rain [Non-Adapt]')

        # Fog (Non-Adaptive) - Red X marker
        plt.plot(PH_RANGE_DBM, mi_fog_fixed, 'rx:', markersize=10, linewidth=1.5, label='Fog [Non-Adapt]')

        plt.xlabel('$P_H$ [dBm]', fontsize=12)
        plt.ylabel('$\mathcal{I}_H$ [bits/H. FSO channel use]', fontsize=12)
        plt.title('Performance of Non-Adaptive Hybrid FSO/RF System', fontsize=14)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.xlim([5, 19])

        plt.tight_layout()
        plt.savefig('figure6_non_adaptive.png')
        plt.show()

        print("\n완료: 'figure6_non_adaptive.png' 저장됨")