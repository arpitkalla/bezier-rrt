# TODO: Start using rtrees for maintaining the graph
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
R = 80. # Radius of curvature in m
V_max = 30. # Max veocity in m/s
a_max = 9.8 # Max acceleration in m/s^2
R_min = V_max**2/a_max # Minimum turning radius
theta_1_set = np.radians(-25 + 5. * np.arange(10)) # In radians
theta_2_set = np.radians(-50 + 10. * np.arange(10)) # In radians

##### Waypoints #####
p_init = np.array([0,0]).reshape(2, 1)
e_init = np.array([0., 1.]).reshape(2, 1)
x_init = (p_init, e_init)
p_goal = np.array([10,10]).reshape(2, 1)
T_arrival = 1.
R_goal = 1.


def _bezier(P, T):
	return lambda t: P[0] + 3*t*(P[1]-P[0])/T + 3*t**2*(P[2]-2*P[1]+P[0])/T**2 + t**3*(3*P[1]-3*P[2]+P[3]-P[0])/T**3

def get_primitive(x, T, theta_1, theta_2):
	(P_i, e_i) = x
	D_1 = R / 3
	P_1 = P_i + D_1 * e_i
	e_i_f = np.array([[np.cos(theta_1), -np.sin(theta_1)],
					  [np.sin(theta_1), np.cos(theta_1)]]).dot(e_i)
	e_f = np.array([[np.cos(theta_2), -np.sin(theta_2)],
					[np.sin(theta_2), np.cos(theta_2)]]).dot(e_i)
	P_f = P_i + R * e_i_f
	P_2 = P_f - D_1 * e_f
	P = [P_i, P_1, P_2, P_f]
	return P, e_f

def gen_curve(x_i, T, theta_1, theta_2):
	P, _ = get_primitive(x_i, T, theta_1, theta_2)
	return _bezier(P, T)

def visualize_motion_prim_set(x_init, R, R_min, T_arrival, theta_1_set, theta_2_set):
	(p_init, e_init) = x_init
	t = np.linspace(0, T_arrival, 100)
	
	circle1 = plt.Circle((p_init[0], p_init[1]), R, color='r', fill=False)
	orth_e_i = np.array([-e_init[1], e_init[0]])
	min_turn_center = p_init - orth_e_i * R_min
	min_turn_1 = plt.Circle((min_turn_center[0], min_turn_center[1]), R_min, color='g', fill=False)
	min_turn_center = p_init + orth_e_i * R_min
	min_turn_2 = plt.Circle((min_turn_center[0], min_turn_center[1]), R_min, color='g', fill=False)
	
	fig, ax = plt.subplots()
	ax.add_artist(circle1)
	ax.add_artist(min_turn_1)
	ax.add_artist(min_turn_2)

	for i in range(len(theta_1_set)):
		c = gen_curve(x_init, T_arrival, theta_1_set[i], theta_2_set[i])
		curve = c(t)
		ax.plot(curve[0], curve[1], "b-")
	plt.show()

# visualize_motion_prim_set(x_init, R, R_min, T_arrival, theta_1_set, theta_2_set)

def get_nearest(V, p_rand):
	# TODO: Convert to be compatible with RTrees for a massive speedup
	shortest = float("Inf")
	for (p, e) in V:
		d = np.linalg.norm(p_rand - p)
		if d < shortest:
			shortest = d
			p_nearest = p
			e_nearest = e
	return (p_nearest, e_nearest)

def steer(p_nearest, p_rand):
	if np.linalg.norm(p_rand - p_nearest) > R:
		return p_nearest + (p_rand - p_nearest)*R/np.linalg.norm(p_rand - p_nearest)
	return p_rand

def get_primitive_set(x):
	X_set = []
	for i in range(len(theta_1_set)):
		P, e_f = get_primitive(x, T_arrival, theta_1_set[i], theta_2_set[i])
		X_set.append((P[-1], e_f))
	return X_set

def get_new_vertex(p_ref, X_set):
	# TODO: This might change with rtree implementation
	return get_nearest(X_set, p_ref)

def is_collision(x_1, x_2):
	# TODO: Obstacle checker
	return False

def extend_tree(G, p_rand):
	V_prime, V_prime_goal, E_prime = G[-1]
	x_nearest = get_nearest(V_prime, p_rand)
	(p_nearest, _) = x_nearest
	if np.linalg.norm(p_nearest - p_goal) < R_goal:
		return (V_prime, V_prime_goal, E_prime)

	p_ref = steer(p_nearest, p_rand)
	X_set = get_primitive_set(x_nearest)
	x_new = get_new_vertex(p_ref, X_set)
	(p_new, _) = x_new
	if not is_collision(x_nearest, x_new):
		V_prime.append(x_new)
		E_prime.append((x_nearest, x_new))
		if np.linalg.norm(p_new - p_goal) < R_goal:
			V_prime_goal.append(x_new)
	return (V_prime, V_prime_goal, E_prime)

def get_sample(i):
	return -10 + 10 * np.random.rand(2,1)

def RRT():
	i = 0
	N_sample = 100
	V = [(p_init, e_init)]
	V_goal = []
	E = []
	G = []

	while i < N_sample:
		G.append((V, V_goal, E))
		p_rand = get_sample(i)
		i += 1
		V, V_goal, E = extend_tree(G, p_rand)

	print(V)
	# J = get_cost(G, X_goal)
	# P_path, V_iner, D_est = get_results(A, G, J, X_goal, T_arrival)

RRT()