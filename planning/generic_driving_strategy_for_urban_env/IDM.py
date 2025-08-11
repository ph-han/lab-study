import math
import matplotlib.pyplot as plt

class IDMVehicle:
    def __init__(self, x=0, v=0, v0=3, T=1.5, a_max=1.0, b_comf=1.5, s0=2.0, length=5.0):
        # --- 차량의 현재 상태 ---
        self.x = x  # 위치 (m)
        self.v = v  # 속도 (m/s)
        self.a = 0.0  # 가속도 (m/s^2)

        # --- IDM 파라미터 ---
        self.v0 = v0  # 목표 속도 (m/s)
        self.T = T  # 안전 시간 간격 (s)
        self.a_max = a_max  # 최대 가속도 (m/s^2)
        self.b_comf = b_comf  # 안락한 감속도 (m/s^2)
        self.s0 = s0  # 정지 시 최소 간격 (m)
        self.length = length  # 차량 길이 (m)
        self.delta = 4.0  # 가속 지수

    def update_acceleration(self, leader=None):
        """앞차(leader) 정보를 바탕으로 자신의 가속도를 업데이트합니다."""
        if leader:
            # 앞차가 있으면 간격(s)과 상대 속도(delta_v) 계산
            # 간격(s) = 앞차 위치 - 내 차 위치 - 앞차 길이
            s = leader.x - self.x - leader.length
            delta_v = self.v - leader.v
        else:
            # 앞차가 없으면 상호작용 항이 0이 되도록 설정
            s = float('inf')
            delta_v = 0

        # 0으로 나누기 방지
        if s < 1e-5: s = 1e-5

        s_star = self.s0 + max(0, self.v * self.T + (self.v * delta_v) / (2 * math.sqrt(self.a_max * self.b_comf)))

        self.a = self.a_max * (1 - (self.v / self.v0) ** self.delta - (s_star / s) ** 2)

    def update_state(self, dt=0.1):
        """시간 스텝(dt)에 따라 차량의 속도와 위치를 업데이트합니다."""
        # 오일러 적분(Euler integration)을 사용한 간단한 업데이트
        self.v += self.a * dt

        # 차량이 후진하지 않도록 속도는 항상 0 이상으로 유지
        if self.v < 0:
            self.v = 0
        self.x += self.v * dt


if __name__ == "__main__":
    leader = IDMVehicle(x=27, v=1)
    follower = IDMVehicle(x=10, v=3)

    total_time = 40
    dt = 0.1
    n_steps = int(total_time / dt)

    time_points = []
    leader_positions = []
    follower_positions = []
    leader_velocities = []
    follower_velocities = []

    # --- 시뮬레이션 루프 (수정 없음) ---
    for step in range(n_steps):
        current_time = step * dt

        if current_time > 10 and leader.v0 < 25:
            leader.v0 = 25

        leader.update_acceleration(leader=None)
        follower.update_acceleration(leader=leader)

        leader.update_state(dt)
        follower.update_state(dt)

        time_points.append(current_time)
        leader_positions.append(leader.x)
        follower_positions.append(follower.x)
        leader_velocities.append(leader.v)
        follower_velocities.append(follower.v)

    # --- 그래프 그리기 (x축과 y축 변경) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 그래프를 좌우로 배치

    # 1. 시간-거리 그래프 (Time vs. Distance)
    ax1.plot(leader_positions, time_points, label='Leader Vehicle', color='blue')  # x축: 위치, y축: 시간
    ax1.plot(follower_positions, time_points, label='Follower Vehicle', color='red', linestyle='--')  # x축: 위치, y축: 시간
    ax1.set_title('Time vs. Distance')
    ax1.set_xlabel('Position (m)')  # x축 레이블 변경
    ax1.set_ylabel('Time (s)')  # y축 레이블 변경
    ax1.legend()
    ax1.grid(True)

    # 2. 시간-속도 그래프 (Time vs. Velocity)
    ax2.plot(leader_velocities, time_points, label='Leader Vehicle', color='blue')  # x축: 속도, y축: 시간
    ax2.plot(follower_velocities, time_points, label='Follower Vehicle', color='red', linestyle='--')  # x축: 속도, y축: 시간
    ax2.set_title('Time vs. Velocity')
    ax2.set_xlabel('Velocity (m/s)')  # x축 레이블 변경
    ax2.set_ylabel('Time (s)')  # y축 레이블 변경
    ax2.legend()
    ax2.grid(True)

    # 그래프 레이아웃 조정 및 표시
    plt.tight_layout()
    plt.show()
