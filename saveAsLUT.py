import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from scipy.stats import gamma, rice
import pickle
from scipy.interpolate import interp1d

"""
OOK부터 16-PAM까지, pointing error 그래프 출력
"""
# ----------------------------------------------------------------
# 1. 채널 생성 함수
# ----------------------------------------------------------------
def generate_pointing_error(xi=1.2, P_eq=1.0, size=1000):
    """
    수식 (21) PDF를 따르는 포인팅 에러 샘플 생성 (Inverse Transform Sampling)

    매개변수:
    xi (float): 등가 포인팅 에러 계수 (xi_eq). 작을수록 에러가 심함. (기본값 1.2)
    P_eq (float): 포인팅 에러가 없을 때의 최대 수신 전력. (Zero Boresight 가정 시 1.0)
    size (int): 생성할 샘플의 개수. (기본값 1000)

    반환값:
    numpy.ndarray: 포인팅 에러 감쇄 계수 (Ip) 샘플 1000개
    """

    # 1. 0과 1 사이의 균등 분포(Uniform Distribution)에서 난수 u 생성
    u = np.random.rand(size)

    # 2. 수식 (21)의 역함수(Inverse CDF)를 이용한 샘플 변환
    # 수식 유도: Ip = P_eq * u^(1 / xi^2)
    Ip = P_eq * (u ** (1.0 / (xi ** 2)))

    return Ip


def generate_gamma_gamma(alpha, beta, size=1000):
    """
    Gamma-Gamma 분포 채널 생성 (E[I]=1)
    """
    hx = gamma.rvs(alpha, scale=1 / alpha, size=size)
    hy = gamma.rvs(beta, scale=1 / beta, size=size)
    return hx * hy


def generate_rician(K_dB, size=1000):
    """
    Rician 분포 채널 생성 (E[h^2]=1)
    """
    K = 10 ** (K_dB / 10.0)
    sigma = np.sqrt(1 / (2 * (K + 1)))
    b = np.sqrt(2 * K)
    return rice.rvs(b, scale=sigma, size=size)


# ----------------------------------------------------------------
# 2. FSO MI 계산 함수 (OOK, 4-PAM)
# ----------------------------------------------------------------
def get_mi_fso_general(h, snr_db, M, n_samples=4000):
    """
    FSO M-PAM에 대한 MI 계산
    M=2 (OOK), M=4 (4-PAM)
    """
    n_channels = len(h)
    lambda_val = 1.0  # Background noise variance (fixed)
    P_avg = 10 ** (snr_db / 10.0)  # Average optical power

    # --- 심볼 레벨 설정 (평균 전력 P 유지) ---
    # S = {0, d, 2d, ..., (M-1)d}
    # E[S] = d * (M-1)/2 = P  => d = 2P / (M-1)
    if M == 1: return 0
    d = (2.0 * P_avg) / (M - 1)
    symbols = np.arange(M) * d  # shape: (M,)

    # 몬테카를로 샘플링을 위해 각 심볼당 샘플 수 할당
    n_per_sym = n_samples // M
    h_vec = h[:, np.newaxis]  # (N_ch, 1)

    y_list = []

    # --- 데이터 생성 (Generate y) ---
    for s in symbols:
        # Noise variance = h*s + lambda (Signal Dependent)
        # s가 0일 때 h*s=0이므로 분산은 lambda
        var = h_vec * s + lambda_val
        std = np.sqrt(var)

        # y ~ N(h*s, var)
        y_s = np.random.normal(loc=h_vec * s, scale=std, size=(n_channels, n_per_sym))
        y_list.append(y_s)

    # 모든 수신 신호 합치기 (전체 모집단 Y)
    y_all = np.hstack(y_list)  # (N_ch, n_samples)

    # --- 확률 밀도 계산 (Calculate PDFs) ---
    # 각 심볼 x_k를 보냈을 때, 현재 수신된 y_all이 나올 확률 P(y | x_k) 계산
    log_probs_x = []  # 리스트에 각 심볼별 로그 확률 저장

    for s in symbols:
        var = h_vec * s + lambda_val  # (N_ch, 1)
        # var를 y_all 크기에 맞게 확장
        var_mat = np.tile(var, (1, n_samples))
        mu_mat = np.tile(h_vec * s, (1, n_samples))

        # Log Gaussian PDF
        lp = -0.5 * np.log(2 * np.pi * var_mat) - ((y_all - mu_mat) ** 2) / (2 * var_mat)
        log_probs_x.append(lp)

    # --- MI 계산 ---
    # 1. P(y) = (1/M) * sum(P(y|x_k))
    # Log domain calculation: log(sum(exp(lp))) - log(M)
    # logsumexp 구현
    log_probs_stack = np.array(log_probs_x)  # (M, N_ch, n_samples)
    max_lp = np.max(log_probs_stack, axis=0)
    # exp(lp - max)로 오버플로우 방지
    sum_exp = np.sum(np.exp(log_probs_stack - max_lp), axis=0)
    log_p_y = max_lp + np.log(sum_exp) - np.log(M)

    # 2. P(y|x) 추출
    # y_all의 앞부분은 sym[0], 그 다음은 sym[1]... 순서로 생성됨
    log_p_y_given_true_x_list = []
    for i in range(M):
        # i번째 심볼에 해당하는 y 구간만 가져옴
        lp = log_probs_x[i][:, i * n_per_sym: (i + 1) * n_per_sym]
        log_p_y_given_true_x_list.append(lp)

    log_p_y_given_true_x = np.hstack(log_p_y_given_true_x_list)

    # 3. MI = Avg( log2( P(y|x) / P(y) ) )
    mi_samples = (log_p_y_given_true_x - log_p_y) / np.log(2)

    return np.mean(np.mean(mi_samples, axis=1))


# ----------------------------------------------------------------
# 3. RF MI 계산 함수 (BPSK, QPSK)
# ----------------------------------------------------------------
def get_mi_rf_general(h, snr_db, mod_type='BPSK', n_samples=4000):
    """
    RF 채널 M-PSK MI 계산 함수
    지원: BPSK, QPSK, 8-PSK, 16-PSK
    """
    n_channels = len(h)
    Es = 10 ** (snr_db / 10.0)  # Symbol Energy
    N0 = 1.0  # Noise PSD (Total Variance)

    h_vec = h[:, np.newaxis]

    # 1. 변조 방식에 따른 M 값 및 복소수 여부 설정
    if mod_type == 'BPSK':
        M = 2
        is_complex = False
    elif mod_type == 'QPSK':
        M = 4
        is_complex = True
    elif mod_type == '8-PSK':
        M = 8
        is_complex = True
    elif mod_type == '16-PSK':
        M = 16
        is_complex = True
    else:
        raise ValueError(f"Unknown modulation type: {mod_type}")

    # 2. 심볼 생성 (Constellation Generation)
    if not is_complex:
        # BPSK: Real axis {-sqrt(Es), +sqrt(Es)}
        symbols = np.array([-np.sqrt(Es), np.sqrt(Es)])
    else:
        # M-PSK: Complex plane, evenly spaced on circle radius sqrt(Es)
        # s_k = sqrt(Es) * exp(j * 2*pi*k / M)
        phases = np.arange(M) * (2 * np.pi / M)

        # (Optional) QPSK의 경우 일반적으로 pi/4 회전된 성상도를 쓰지만,
        # AWGN/Fading 환경에서 MI는 회전 불변(Rotation Invariant)이므로 0도부터 시작해도 무방합니다.
        # 여기서는 일반화된 수식을 사용합니다.
        symbols = np.sqrt(Es) * np.exp(1j * phases)

    # 각 심볼당 할당할 샘플 수
    n_per_sym = n_samples // M
    y_list = []

    # 3. 데이터 생성 (Generate Received Signals)
    for s in symbols:
        # 잡음 생성
        if is_complex:
            # Complex Gaussian Noise: CN(0, N0)
            # Real Var = N0/2, Imag Var = N0/2
            noise_scale = np.sqrt(N0 / 2.0)
            n_real = np.random.normal(0, noise_scale, (n_channels, n_per_sym))
            n_imag = np.random.normal(0, noise_scale, (n_channels, n_per_sym))
            noise = n_real + 1j * n_imag
        else:
            # Real Gaussian Noise: N(0, N0)
            noise = np.random.normal(0, np.sqrt(N0), (n_channels, n_per_sym))

        # y = h*s + n
        y_s = h_vec * s + noise
        y_list.append(y_s)

    # 전체 수신 신호 (모집단)
    y_all = np.hstack(y_list)

    # 4. 확률 밀도 계산 (Calculate Log-Likelihoods)
    log_probs_x = []

    for s in symbols:
        mu = h_vec * s  # Broadcasting (N_ch, 1) to (N_ch, total_samples)

        if is_complex:
            # Complex Gaussian PDF: (1/(pi*N0)) * exp(-|y-mu|^2 / N0)
            # Log domain: -log(pi*N0) - |y-mu|^2 / N0
            dist_sq = np.abs(y_all - mu) ** 2
            lp = -np.log(np.pi * N0) - dist_sq / N0
        else:
            # Real Gaussian PDF: (1/sqrt(2*pi*N0)) * exp(-(y-mu)^2 / (2*N0))
            dist_sq = (y_all - mu) ** 2
            lp = -0.5 * np.log(2 * np.pi * N0) - dist_sq / (2 * N0)

        log_probs_x.append(lp)

    # 5. MI 계산 (LogSumExp Trick)
    # P(y) = (1/M) * sum(exp(lp))
    log_probs_stack = np.array(log_probs_x)  # Shape: (M, N_ch, n_samples)
    max_lp = np.max(log_probs_stack, axis=0)
    sum_exp = np.sum(np.exp(log_probs_stack - max_lp), axis=0)

    # log(P(y)) = max_lp + log(sum_exp) - log(M)
    log_p_y = max_lp + np.log(sum_exp) - np.log(M)

    # P(y | True Symbol) 추출
    # y_all의 데이터 순서가 심볼 순서와 같으므로 슬라이싱으로 매칭
    log_p_y_given_true_list = []
    for i in range(M):
        # i번째 심볼이 전송되었을 때의 log P(y|x_i)
        lp = log_probs_x[i][:, i * n_per_sym: (i + 1) * n_per_sym]
        log_p_y_given_true_list.append(lp)

    log_p_y_given_true = np.hstack(log_p_y_given_true_list)

    # MI = E[ log2( P(y|x) / P(y) ) ]
    # 자연로그(ln) 차이를 log2로 변환
    mi_samples = (log_p_y_given_true - log_p_y) / np.log(2)

    return np.mean(np.mean(mi_samples, axis=1))


# ----------------------------------------------------------------
# 4. 메인 시뮬레이션 실행
# ----------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # 파라미터
    alpha, beta = 1, 2
    K_dB = 4
    num_channels = 1000
    xi_val = 1.2  # 포인팅 에러 계수 (1.2: 강한 에러, 6.7: 약한 에러)

    print("1. 채널 생성 중...")
    samples_turbulence = generate_gamma_gamma(alpha, beta, size=num_channels)
    samples_pointing = generate_pointing_error(xi=xi_val, size=num_channels)
    h_rf = generate_rician(K_dB, size=num_channels)
    h_fso = samples_turbulence * samples_pointing

    # 테스트용 빠른 버전
    # snr_dbs = np.arange(-20, 51, 10)
    # 진짜 시뮬용
    snr_dbs = np.arange(-20, 51, 2)

    # 결과 저장용 리스트
    mi_fso_ook_pointing_error = []
    mi_fso_4pam_pointing_error = []
    mi_fso_8pam_pointing_error = []
    mi_fso_16pam_pointing_error = []
    mi_rf_bpsk = []
    mi_rf_qpsk = []
    mi_rf_8psk = []
    mi_rf_16psk = []
    snr_axis = np.arange(-20, 51, 2)

    print("2. MI 계산 중 (OOK, 4-PAM, BPSK, QPSK)...")
    for snr in snr_dbs:
        # FSO with pointing error
        mi_fso_ook_pointing_error.append(get_mi_fso_general(h_fso, snr, M=2))
        mi_fso_4pam_pointing_error.append(get_mi_fso_general(h_fso, snr, M=4))
        mi_fso_8pam_pointing_error.append(get_mi_fso_general(h_fso, snr, M=8))
        mi_fso_16pam_pointing_error.append(get_mi_fso_general(h_fso, snr, M=16))

        # RF
        mi_rf_bpsk.append(get_mi_rf_general(h_rf, snr, mod_type='BPSK'))
        mi_rf_qpsk.append(get_mi_rf_general(h_rf, snr, mod_type='QPSK'))
        mi_rf_8psk.append(get_mi_rf_general(h_rf, snr, mod_type='8-PSK'))
        mi_rf_16psk.append(get_mi_rf_general(h_rf, snr, mod_type='16-PSK'))
        print(snr, sep="", end=" ")

    mi_luts = {
        'OOK': interp1d(snr_axis, mi_fso_ook_pointing_error, kind='cubic', fill_value="extrapolate"),
        '4-PAM': interp1d(snr_axis, mi_fso_4pam_pointing_error, kind='cubic', fill_value="extrapolate"),
        '8-PAM': interp1d(snr_axis, mi_fso_8pam_pointing_error, kind='cubic', fill_value="extrapolate"),
        '16-PAM': interp1d(snr_axis, mi_fso_16pam_pointing_error, kind='cubic', fill_value="extrapolate"),

        'BPSK': interp1d(snr_axis, mi_rf_bpsk, kind='cubic', fill_value="extrapolate"),
        'QPSK': interp1d(snr_axis, mi_rf_qpsk, kind='cubic', fill_value="extrapolate"),
        '8PSK': interp1d(snr_axis, mi_rf_8psk, kind='cubic', fill_value="extrapolate"),
        '16PSK': interp1d(snr_axis, mi_rf_16psk, kind='cubic', fill_value="extrapolate"),
    }

    # 그래프 그리기
    plt.figure(figsize=(10, 7))

    # RF Curves (Solid lines)
    plt.plot(snr_dbs, mi_rf_bpsk, 'r-', label='BPSK', linewidth=2)
    plt.plot(snr_dbs, mi_rf_qpsk, color='saddlebrown', label='QPSK', linewidth=2)
    plt.plot(snr_dbs, mi_rf_8psk, 'b-', label='8-PSK', linewidth=2)
    plt.plot(snr_dbs, mi_rf_16psk, 'k-', label='16-PSK', linewidth=2)

    plt.plot(snr_dbs, mi_fso_ook_pointing_error, 'r:', label='OOK with pointing error', linewidth=2)
    plt.plot(snr_dbs, mi_fso_4pam_pointing_error, color='saddlebrown', linestyle='dotted', label='4-PAM with pointing error', linewidth=2)
    plt.plot(snr_dbs, mi_fso_8pam_pointing_error, 'b:', label='8-PAM with pointing error', linewidth=2)
    plt.plot(snr_dbs, mi_fso_16pam_pointing_error, 'k:', label='16-PAM with pointing error', linewidth=2)

    plt.xlabel('P/$\lambda$, $E_s/N_0$ [dB]', fontsize=12)
    plt.ylabel('MI [bits/channel use]', fontsize=12)
    # plt.title('MI vs SNR (Reproduction of Fig 2)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xlim([-20, 50])
    plt.ylim([0, 4.2])  # QPSK goes up to 2 bits

    plt.show()

    with open('mi_lookup_tables.pkl', 'wb') as f:
        pickle.dump(mi_luts, f)

    print("모든 변조 방식의 LUT가 mi_lookup_tables.pkl에 저장되었습니다.")