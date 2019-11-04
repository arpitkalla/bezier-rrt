import numpy as np
import matplotlib.pyplot as plt
##### Aircraft Specification ######
m = 1. # Total mass
S_ref = 1. # Reference Area
C_L_0 = 1. # Lift coefficient at zero ange of attack
C_L_alpha = 1. # Lift curve slope
C_D_0 = 1. # Drag coefficient at zero lift coefficient
K = 1. # (C_D - C_D_0) / C_L^2
A = (m, S_ref, C_L_0, C_L_alpha, C_D_0, K)

##### Motion Primitive Set #####
R = 1. # Radius of curvature in m
V_max = 30. # Max veocity in m/s
A_max = 9.8 # Max acceleration in m/s^2
R_turn = V_max**2/A_max
theta_1_set = np.radians(-25 + 5 * np.arange(10)) # In degrees
theta_2_set = np.radians(-50 + 10 * np.arange(10)) # In degrees

##### Waypoints #####
X_init = np.array([1,1]).reshape(2, 1)
e_i = np.array([0., 1.]).reshape(2, 1)
X_goal = np.array([10,10]).reshape(2, 1)
T_arrival = 1.


def bezier(P, T):
	return lambda t: P[0] + 3*t*(P[1]-P[0])/T + 3*t**2*(P[2]-2*P[1]+P[0])/T**2 + t**3*(3*P[1]-3*P[2]+P[3]-P[0])/T**3

def gen_curve(P_i, e_i, T, theta_1, theta_2):
	D_1 = R / 3
	P_1 = P_i + D_1 * e_i
	e_i_f = np.array([[np.cos(theta_1), -np.sin(theta_1)],
					  [np.sin(theta_1), np.cos(theta_1)]]).dot(e_i)
	e_f = np.array([[np.cos(theta_2), -np.sin(theta_2)],
					[np.sin(theta_2), np.cos(theta_2)]]).dot(e_i)
	P_f = P_i + R * e_i_f
	P_2 = P_f - D_1 * e_f
	P = [P_i, P_1, P_2, P_f]
	return bezier(P, T)

curve = gen_curve(X_init, e_i, T_arrival, theta_1_set[0], theta_2_set[0])
def visualize_motion_prim_set(X_init, e_i, R, T_arrival, theta_1_set, theta_2_set):
	t = np.linspace(0, T_arrival, 100)
	fig, ax = plt.subplots()
	circle1 = plt.Circle((X_init[0], X_init[1]), R, color='b', fill=False)
	ax.add_artist(circle1)
	for i in range(len(theta_1_set)):
		c = gen_curve(X_init, e_i, T_arrival, theta_1_set[i], theta_2_set[i])
		curve = c(t)
		ax.plot(curve[0], curve[1], "r.")
	
	plt.show()

visualize_motion_prim_set(X_init, e_i, R, T_arrival, theta_1_set, theta_2_set)

def RRT():
	i = 0
	N_sample = 100
	V = set(X_init)
	V_goal = set()
	E = set()
	G = []

	while i < N_sample:
		G.append((V, V_goal, E))
		p_rand = get_sample(i)
		i += 1
		V, V_goal, E = extend_tree(G, p_rand)

	J = get_cost(G, X_goal)
	P_path, V_iner, D_est = get_results(A, G, J, X_goal, T_arrival)