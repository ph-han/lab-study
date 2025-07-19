'''
Made by gemini-2.5-pro
'''

import numpy as np
import math
from Car import Car
from scipy.optimize import minimize

K_MAX = math.tan(Car.MAX_STEER) / Car.WHEEL_BASE

# --- 비용 및 그래디언트 계산 함수 ---

def calculate_cost(path_flat, obstacles, w_s, w_k, w_o, k_max, d_max):
    """경로의 전체 비용(목적 함수)을 계산합니다."""
    path = path_flat.reshape(-1, 2)
    N = len(path)
    
    cost_s = 0.0  # Smoothness cost
    cost_k = 0.0  # Curvature cost
    cost_o = 0.0  # Obstacle cost

    for i in range(1, N - 1):
        # 1. 부드러움 비용 (Smoothness Cost)
        smoothness_vector = path[i-1] - 2 * path[i] + path[i+1]
        cost_s += np.dot(smoothness_vector, smoothness_vector)

        # 2. 곡률 비용 (Curvature Cost)
        v_prev = path[i] - path[i-1]
        v_next = path[i+1] - path[i]
        mag_prev = np.linalg.norm(v_prev)
        mag_next = np.linalg.norm(v_next)
        
        if mag_prev > 1e-6 and mag_next > 1e-6:
            cos_delta_phi = np.dot(v_prev, v_next) / (mag_prev * mag_next)
            cos_delta_phi = np.clip(cos_delta_phi, -1.0, 1.0)
            delta_phi = np.arccos(cos_delta_phi)
            kappa = delta_phi / mag_prev
            
            if kappa > k_max:
                cost_k += (kappa - k_max)**2

    # 3. 장애물 비용 (Obstacle Cost)
    if obstacles.size > 0:
        for i in range(N):
            p = path[i]
            dist_sq = np.sum((obstacles - p)**2, axis=1)
            min_dist = np.sqrt(np.min(dist_sq))
            
            if min_dist < d_max:
                cost_o += (min_dist - d_max)**2
            
    total_cost = w_s * cost_s + w_k * cost_k + w_o * cost_o
    return total_cost


def calculate_gradient(path_flat, obstacles, w_s, w_k, w_o, k_max, d_max):
    """경로 비용의 그래디언트를 계산합니다."""
    path = path_flat.reshape(-1, 2)
    N = len(path)
    
    # 각 비용 요소에 대한 그래디언트를 개별적으로 계산
    gradient_s = np.zeros_like(path)
    gradient_k = np.zeros_like(path)
    gradient_o = np.zeros_like(path)

    # 1. 부드러움 그래디언트
    for i in range(1, N - 1):
        grad_vec = 2 * (path[i-1] - 2 * path[i] + path[i+1])
        gradient_s[i-1] += grad_vec
        gradient_s[i]   -= 2 * grad_vec
        gradient_s[i+1] += grad_vec

    # 2. 곡률 그래디언트
    for i in range(1, N - 1):
        p_im1, p_i, p_ip1 = path[i-1], path[i], path[i+1]
        delta_x_i = p_i - p_im1
        delta_x_ip1 = p_ip1 - p_i
        l_i = np.linalg.norm(delta_x_i)
        l_ip1 = np.linalg.norm(delta_x_ip1)

        if l_i < 1e-6 or l_ip1 < 1e-6: continue

        cos_phi = np.clip(np.dot(delta_x_i, delta_x_ip1) / (l_i * l_ip1), -1.0, 1.0)
        phi = np.arccos(cos_phi)
        kappa = phi / l_i

        if kappa > k_max:
            sin_phi = np.sin(phi)
            if sin_phi < 1e-6: continue
            
            p1_num = delta_x_i - (np.dot(delta_x_i, delta_x_ip1) / (l_ip1**2)) * delta_x_ip1
            p1 = p1_num / (l_i * l_ip1)
            p2_num = delta_x_ip1 - (np.dot(delta_x_i, delta_x_ip1) / (l_i**2)) * delta_x_i
            p2 = p2_num / (l_i * l_ip1)

            d_phi_d_pim1 = -1/sin_phi * p2
            d_phi_d_pi   = -1/sin_phi * (-p1 - p2)
            d_phi_d_pip1 = -1/sin_phi * p1

            d_li_d_pim1 = -delta_x_i / l_i
            d_li_d_pi   = delta_x_i / l_i

            grad_kappa_pim1 = (1/l_i) * d_phi_d_pim1 - (phi/(l_i**2)) * d_li_d_pim1
            grad_kappa_pi   = (1/l_i) * d_phi_d_pi   - (phi/(l_i**2)) * d_li_d_pi
            grad_kappa_pip1 = (1/l_i) * d_phi_d_pip1

            common_factor = 2 * (kappa - k_max)
            gradient_k[i-1] += common_factor * grad_kappa_pim1
            gradient_k[i]   += common_factor * grad_kappa_pi
            gradient_k[i+1] += common_factor * grad_kappa_pip1

    # 3. 장애물 그래디언트
    if obstacles.size > 0:
        for i in range(N):
            p = path[i]
            dist_sq = np.sum((obstacles - p)**2, axis=1)
            min_idx = np.argmin(dist_sq)
            o_nearest = obstacles[min_idx]
            dist = np.sqrt(dist_sq[min_idx])

            if dist < d_max and dist > 1e-6:
                grad_o_vec = 2 * (dist - d_max) * (p - o_nearest) / dist
                gradient_o[i] += grad_o_vec
    
    # 최종 그래디언트 (가중치 적용)
    gradient = w_s * gradient_s + w_k * gradient_k + w_o * gradient_o
    return gradient.flatten()


# --- 메인 함수 (외부에서 호출 가능) ---

def smooth_path(initial_path, obstacles, w_s=10.0, w_k=15.0, w_o=5.0, k_max=K_MAX, d_max=1.5, max_iter=100):
    """
    켤레 기울기법(Conjugate Gradient)을 사용하여 경로를 부드럽게 만듭니다.

    Args:
        initial_path (np.array): 초기 경로 (N, 2)
        obstacles (np.array): 장애물 위치 (M, 2)
        w_s (float): 부드러움(smoothness) 가중치
        w_k (float): 곡률(curvature) 가중치
        w_o (float): 장애물(obstacle) 가중치
        k_max (float): 최대 허용 곡률
        d_max (float): 장애물 최소 안전 거리
        max_iter (int): 최대 반복 횟수

    Returns:
        np.array: 부드러워진 경로 (N, 2)
    """
    # Scipy의 최적화 함수 호출
    result = minimize(
        fun=calculate_cost,
        x0=initial_path.flatten(),
        args=(obstacles, w_s, w_k, w_o, k_max, d_max),
        method='CG',
        jac=calculate_gradient,
        options={'maxiter': max_iter, 'disp': False} # 라이브러리로 사용할 때는 disp=False가 유용
    )

    smoothed_path = result.x.reshape(-1, 2)
    return smoothed_path[:, 0].tolist(), smoothed_path[:, 1].tolist()
