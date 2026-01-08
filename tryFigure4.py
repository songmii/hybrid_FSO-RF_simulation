import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d

# ----------------------------------------------------------------
# 1. 설정 및 초기화
# ----------------------------------------------------------------

# 논문 및 요청에 따른 파라미터 설정
RHO = 0.3  # FSO 채널 기상 계수 (Foggy conditions 가정)
RHO_HAT = 0.9  # RF 채널 기상 계수
BETA = 1.0  # 논문 Table 1, 2 기준 (30dB -> 1000)
BETA_HAT = 1.0  # 논문 Table 1, 2 기준 (30dB -> 1000)
POPULATION_SIZE = 20
MAX_ITERATIONS = 50

# 변조 방식 매핑 (lookup table 키와 일치)
FSO_MODS = {1: 'OOK', 2: '4-PAM', 3: '8-PAM', 4: '16-PAM'}
RF_MODS = {1: 'BPSK', 2: 'QPSK', 3: '8PSK', 4: '16PSK'}


def load_lookup_table(filename='mi_lookup_tables.pkl'):
    """pkl 파일에서 MI Lookup Table을 불러옵니다."""
    try:
        with open(filename, 'rb') as f:
            luts = pickle.load(f)
        print(f"'{filename}' 로드 성공.")
        return luts
    except FileNotFoundError:
        print(f"오류: '{filename}'을 찾을 수 없습니다. saveAsLUT.py를 먼저 실행해주세요.")
        return None


# ----------------------------------------------------------------
# 2. 유틸리티 함수 (수식 구현)
# ----------------------------------------------------------------

def get_rf_snr_linear(p_opt_linear, p_h_linear, m, m_hat):
    """
    수식 (23) JPC (Joint Power Constraint) 구현
    Total Power(P_H)와 Optical SNR(P/lambda)가 주어졌을 때,
    남은 전력으로 가능한 RF SNR(Es/N0)을 계산.

    모든 입력은 dB가 아닌 Linear 스케일이어야 함.
    """
    # 식 (23): Es/N0 = rho_hat * P_H * beta_hat * (m + m_hat) - (P/lambda) * ( (rho_hat * beta_hat) / (rho * beta) )

    term1 = RHO_HAT * p_h_linear * BETA_HAT * (m + m_hat)
    term2 = p_opt_linear * ((RHO_HAT * BETA_HAT) / (RHO * BETA))

    rf_snr_lin = term1 - term2
    return rf_snr_lin


def get_mi(luts, snr_db, mod_idx, channel_type='FSO'):
    """Lookup Table을 이용해 MI 반환"""
    if channel_type == 'FSO':
        key = FSO_MODS.get(int(mod_idx))
    else:
        key = RF_MODS.get(int(mod_idx))

    if key is None:
        return 0.0

    # LUT 범위 밖 처리 (Extrapolation은 LUT 생성 시 설정됨)
    try:
        mi = luts[key](snr_db)
        # MI는 0보다 작을 수 없고 log2(M)보다 클 수 없음
        max_mi = np.log2(2 ** int(mod_idx) if channel_type == 'FSO' and int(mod_idx) > 1 else (
            2 if int(mod_idx) == 1 else int(mod_idx) * 1))  # OOK=1bit, others log2(M).
        # LUT 키가 변조 차수가 아니라 이름이라 별도 처리 필요하지만
        # 간단히 interp1d 결과 사용 (LUT가 이미 정확하다고 가정)
        return float(mi)
    except Exception:
        return 0.0


def calculate_fitness(position, p_h_linear, luts):
    """
    입자(Particle)의 위치에 따른 총 MI(Fitness) 계산.
    position = [P_opt_dB, m, m_hat]
    """
    p_opt_db = position[0]
    m = int(round(position[1]))  # 정수로 반올림 (1~4)
    m_hat = int(round(position[2]))  # 정수로 반올림 (1~4)

    # 범위 제한 (1~4)
    m = np.clip(m, 1, 4)
    m_hat = np.clip(m_hat, 1, 4)

    # 1. Optical SNR (Linear) 변환
    p_opt_linear = 10 ** (p_opt_db / 10.0)

    # 2. RF SNR 계산 (JPC 제약 조건 적용)
    rf_snr_linear = get_rf_snr_linear(p_opt_linear, p_h_linear, m, m_hat)

    # 3. 전력 제약 조건 위배 여부 확인
    # RF에 할당할 전력이 음수가 되면(FSO가 너무 많이 쓰면) 유효하지 않은 해
    if rf_snr_linear <= 0:
        return -1.0  # 페널티 부여

    rf_snr_db = 10 * np.log10(rf_snr_linear)

    # 4. MI Lookup
    # m=1 (OOK) -> FSO_MODS[1], m=1 (BPSK) -> RF_MODS[1]
    # 실제 m값은 bits/symbol이 아니라 index (1,2,3,4)로 사용됨을 가정 (saveAsLUT.py 참조)
    # 하지만 saveAsLUT에서 M=2,4,8,16을 썼으므로,
    # FSO: OOK(idx 1, M=2), 4PAM(idx 2, M=4)...
    # m과 m_hat은 논문상 'bits per symbol'이 아니라 'mode index'로 근사화하여 최적화 수행
    # (논문에서는 m이 bits 수인지 index인지 문맥에 따라 다르나, 시뮬레이션 편의상 index 1~4로 최적화하고 LUT 매핑)

    mi_fso = get_mi(luts, p_opt_db, m, channel_type='FSO')
    mi_rf = get_mi(luts, rf_snr_db, m_hat, channel_type='RF')

    total_mi = mi_fso + mi_rf
    return total_mi


# ----------------------------------------------------------------
# 3. APSO 알고리즘
# ----------------------------------------------------------------

def run_apso_simulation(luts):
    # P_H 범위 (4 dBm ~ 28 dBm)
    p_h_range_dbm = np.arange(4, 29, 1)  # 4 to 28

    # 결과 저장용 리스트
    optimal_p_opt_db_list = []
    max_mi_list = []

    print(f"APSO 시뮬레이션 시작 (P_H: 4~28 dBm)...")

    for p_h_dbm in p_h_range_dbm:
        p_h_linear = 10 ** (p_h_dbm / 10.0)  # dBm to Linear (Assuming normalized unit, or mW consistent with beta)

        # 2-1. 초기화
        # 입자 위치: [P_opt_dB, m, m_hat]
        # P_opt_dB 범위: -10 ~ 40 (탐색 범위)
        # m, m_hat 범위: 1 ~ 4

        # 입자 위치 초기화
        pos_p_opt = np.random.uniform(-10, 40, POPULATION_SIZE)
        pos_m = np.random.uniform(1, 4.99, POPULATION_SIZE)
        pos_m_hat = np.random.uniform(1, 4.99, POPULATION_SIZE)

        particles = np.stack((pos_p_opt, pos_m, pos_m_hat), axis=1)

        # 속도 초기화 (0.1 * p_i)
        velocities = 0.1 * particles

        # pbest, gbest 초기화
        pbest_pos = particles.copy()
        pbest_val = np.full(POPULATION_SIZE, -np.inf)

        gbest_pos = np.zeros(3)
        gbest_val = -np.inf

        # 2-2. 초기 평가
        for i in range(POPULATION_SIZE):
            fit = calculate_fitness(particles[i], p_h_linear, luts)
            pbest_val[i] = fit
            if fit > gbest_val:
                gbest_val = fit
                gbest_pos = particles[i].copy()

        # APSO 루프
        w_max = 0.9
        w_min = 0.4
        c1 = 2.0
        c2 = 2.0

        for t in range(MAX_ITERATIONS):
            # 2-4. omega 계산 (선형 감소)
            omega = w_max - ((w_max - w_min) * t / MAX_ITERATIONS)

            # 랜덤 벡터 생성
            eps1 = np.random.rand(POPULATION_SIZE, 3)
            eps2 = np.random.rand(POPULATION_SIZE, 3)

            # 2-5. 속도 및 위치 업데이트
            # v^(t+1) = w*v^t + c1*eps1*(pbest - p) + c2*eps2*(gbest - p)
            velocities = (omega * velocities +
                          c1 * eps1 * (pbest_pos - particles) +
                          c2 * eps2 * (gbest_pos - particles))  # gbest broadcasting

            particles = particles + velocities

            # 2-6. 제약 조건 확인 및 경계 처리
            # m, m_hat은 1~4 사이여야 함
            particles[:, 1] = np.clip(particles[:, 1], 1, 4.99)
            particles[:, 2] = np.clip(particles[:, 2], 1, 4.99)
            # P_opt_dB는 너무 낮거나 높지 않게 대략적 클리핑 (수렴 돕기 위해)
            particles[:, 0] = np.clip(particles[:, 0], -20, 50)

            # 2-7. 평가 및 업데이트
            for i in range(POPULATION_SIZE):
                current_val = calculate_fitness(particles[i], p_h_linear, luts)

                # Pbest 업데이트
                if current_val > pbest_val[i]:
                    pbest_val[i] = current_val
                    pbest_pos[i] = particles[i].copy()

                # Gbest 업데이트
                if current_val > gbest_val:
                    gbest_val = current_val
                    gbest_pos = particles[i].copy()

        # 루프 종료 후 결과 저장
        # gbest_pos[0]이 최적의 P/lambda (dB)
        optimal_p_opt_db_list.append(gbest_pos[0])
        max_mi_list.append(gbest_val)


        print(f"P_H: {p_h_dbm} dBm -> Max MI: {gbest_val:.4f}, Optimal P/λ: {gbest_pos[0]:.2f} dB")


    return p_h_range_dbm, optimal_p_opt_db_list, max_mi_list


# ----------------------------------------------------------------
# 4. 메인 실행 및 그래프 출력
# ----------------------------------------------------------------

if __name__ == "__main__":
    # 1. pkl 파일 읽어오기
    luts = load_lookup_table()

    if luts:
        # 2. APSO 실행
        ph_axis, opt_snr_axis, mi_axis = run_apso_simulation(luts)

        # 3. 그래프 그리기 (Figure 4 (a) & (b))
        plt.figure(figsize=(12, 5))

        # (a) Optimal P/lambda vs P_H
        plt.subplot(1, 2, 1)
        plt.plot(ph_axis, opt_snr_axis, 'kd--', markerfacecolor='none', label=f'ρ={RHO}, $\hat{{ρ}}$={RHO_HAT}')
        plt.xlabel('$P_H$ [dBm]', fontsize=12)
        plt.ylabel('Optimal $P/\lambda$ [dB]', fontsize=12)
        plt.title('(a) Optimal $P/\lambda$ for given $P_H$', fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()

        # (b) Best Total MI vs P_H
        plt.subplot(1, 2, 2)
        plt.plot(ph_axis, mi_axis, 'rx--', markersize=8, label=f'ρ={RHO}, $\hat{{ρ}}$={RHO_HAT}')
        plt.xlabel('$P_H$ [dBm]', fontsize=12)
        plt.ylabel('$\mathcal{I}_H$ [bits/H.channel use]', fontsize=12)
        plt.title('(b) Best $\mathcal{I}_H$ at optimal $P/\lambda$', fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()

        plt.tight_layout()
        plt.savefig('tryFigure4.png')
        plt.show()
        print("시뮬레이션 완료: figure4_reproduction.png 저장됨")