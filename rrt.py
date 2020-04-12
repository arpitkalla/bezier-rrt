# TODO: Start using rtrees for maintaining the graph
import time
import bezier
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from graph import Graph
##### Aircraft Specification ######
AircraftParams = namedtuple('Aircraft_Params', 
	['mass', 'Sref', 'Cl_0', 'Cl_alpha', 'CD_0', 'K'])
m = 1. # Total mass
S_ref = 1. # Reference Area
C_L_0 = 1. # Lift coefficient at zero ange of attack
C_L_alpha = 1. # Lift curve slope
C_D_0 = 1. # Drag coefficient at zero lift coefficient
K = 1. # (C_D - C_D_0) / C_L^2
A = AircraftParams(m, S_ref, C_L_0, C_L_alpha, C_D_0, K)

##### Motion Primitive Set #####
V_max = 25. # Max veocity in m/s
a_max = 9.81 # Max acceleration in m/s^2
R_min = V_max**2/a_max # Minimum turning radius
R_max = 500
theta_1_set = np.radians(-25 + 5. * np.arange(10)) # In radians
theta_2_set = np.radians(-50 + 10. * np.arange(10)) # In radians
N_prim = 3
R_set = np.arange(R_min, R_max, step=20)


##### Waypoints #####
p_init = np.array([0,0]).reshape(2, 1)
e_init = np.array([0., 1.]).reshape(2, 1)
X_init = (p_init, e_init)
p_goal = np.array([400,500.]).reshape(2, 1)
X_goal = (p_goal, e_init)
T_arrival = 10.
R_goal = 90.
D_1 = V_max #R_prim / 3

##### Cost #####
c1 = 1.
c2 = 1.
c3 = 1.

##### Globals and Constants #####
N_sample = 1000
bias_prob = 0.6
obstacles = [
	(80,300,150), 
	(200,10,150) ]

def _choose_best_path(path_list):
	best_path = None
	best_i = -1
	shortest_distance = float('Inf')
	for idx, path in enumerate(path_list):
		distance = 0
		for i in range(len(path) - 2):
			X_1 = t2x(path[i])
			X_2 = t2x(path[i + 1])
			distance += _get_dist(X_1[0], X_2[0])
		if distance <= shortest_distance:
			best_i = idx
			best_path = path
			shortest_distance = distance
	return best_path


# Algorithm 1 from the paper
def main():
	"""
		A: AircraftParams
		X_init: (p_init, e_init)
		X_goal: (p_goal, e_goal)
		T_arrival: int - Time taken to arive at goal
	"""
	start = time.time()
	V = {x2t(X_init)}
	V_goalarea = set()
	E = set()
	G = (V, V_goalarea, E)
	for i in range(N_sample):
		G = (V, V_goalarea, E)
		p_rand = get_sample(i)
		V, V_goalarea, E = extend_tree(G, p_rand)

	paths_list = []
	for V_ga in V_goalarea:
		paths= get_paths(V, E, V_ga)
		for p in paths:
			if len(p) > 0:
				paths_list.append(p)

	best_path = _choose_best_path(paths_list)
	print("Total time taken with {}: {}".format(N_sample, time.time() - start))

	if best_path is None:
		print("Couldn't find any path.")
		return
	print("Found {} paths.".format(len(paths_list)))
	print("PLOTTING")
	fig, ax = plt.subplots()
	visualize_G(ax, G, path=best_path)
	visualize_obstacles(ax, obstacles)
	plt.axis("equal")
	plt.show()
	# J = get_cost(G, X_goal)
	# P_path, V_iner, D_est = get_results(A, G, J, X_goal, T_arrival)

def _get_bezier_curve(x1, x2):
	pi, ei = x1
	pf, ef = x2
	p1 = pi + D_1 * ei
	p2 = pf - D_1 * ef
	nodes = np.concatenate([pi, p1, p2, pf], axis=1)
	return bezier.Curve(nodes.T, degree=4)


def visualize_splines(ax, E, color="r", alpha=0.3):
	for (t1, t2) in E:
		curve = _get_bezier_curve(t2x(t1), t2x(t2))
		curve.plot(20, color=color,  alpha=alpha, ax=ax)

def visualize_obstacles(ax, obstacles, color="orange", alpha=0.7):
	for (x, y, r) in obstacles:
		circle = plt.Circle((x, y), r, color=color, alpha=alpha, fill=True)
		ax.add_artist(circle)

def visualize_primitive_set():
	(p_init, e_init) = X_init
	t = np.linspace(0, T_arrival, 100)
	
	fig, ax = plt.subplots()

	for R in R_set:
		circle1 = plt.Circle((p_init[0], p_init[1]), R, color='r', fill=False)
		ax.add_artist(circle1)

	orth_e_i = np.array([-e_init[1], e_init[0]])
	min_turn_center = p_init - orth_e_i * R_min
	min_turn_1 = plt.Circle((min_turn_center[0], min_turn_center[1]), R_min, color='g', fill=False)
	min_turn_center = p_init + orth_e_i * R_min
	min_turn_2 = plt.Circle((min_turn_center[0], min_turn_center[1]), R_min, color='g', fill=False)
	ax.add_artist(min_turn_1)
	ax.add_artist(min_turn_2)

	X_set = get_primitive_set(X_init)
	E = map(lambda X: (x2t(X_init), X), X_set)
	visualize_splines(ax, E)

	plt.axis("equal")
	plt.gca().set_xlim(-R_min, R_min)
	plt.show()

def visualize_G(ax, G, path):
	V, V_goalarea, E = G
	points = []
	for (p, e) in V:
		points.append(p)
	points = np.asarray(points)
	ax.plot(points[:, 0], points[:, 1], "r.", alpha=0.3)
	ax.plot(p_init[0], p_init[1], "go")
	ax.plot(p_goal[0], p_goal[1], "bo")

	E_path = set()
	for i in range(len(path) - 2):
		E_path.add((path[i], path[i+1]))
	E_path.add((path[len(path) - 2], x2t(X_goal)))

	points = []
	for (p, e) in path[:-1]:
		points.append(p)
	points.append(x2t(X_goal)[0])
	points = np.asarray(points)
	ax.plot(points[:, 0], points[:, 1], "g.")

	visualize_splines(ax, E)
	visualize_splines(ax, E_path, color="g", alpha=1)
	

# Equation (2) from the paper
def _bezier(P, T):
	"""
		P: List(np.ndarray) - [p_init, control_p1, control_p2, p_final]
		T: int - Travel time
	"""
	return lambda t: P[0] + 3*t*(P[1]-P[0])/T + 3*t**2*(P[2]-2*P[1]+P[0])/T**2 + \
					 t**3*(3*P[1]-3*P[2]+P[3]-P[0])/T**3

def _get_primitive(X, R_prim, theta_1, theta_2):
	(P_i, e_i) = X
	
	P_1 = P_i + D_1 * e_i
	e_to_f = np.array([[np.cos(theta_1), -np.sin(theta_1)],
					  [np.sin(theta_1), np.cos(theta_1)]]).dot(e_i)
	e_f = np.array([[np.cos(theta_2), -np.sin(theta_2)],
					[np.sin(theta_2), np.cos(theta_2)]]).dot(e_i)
	
	P_f = P_i + R_prim * e_to_f
	P_2 = P_f - D_1 * e_f

	return [P_i, P_1, P_2, P_f], e_f

def get_primitive_set(X):
	X_set = set()
	for R in R_set:
		for i in range(len(theta_1_set)):
			P, e = _get_primitive(X, R, theta_1_set[i], theta_2_set[i])
			X_set.add(x2t((P[-1], e)))
	return X_set


def x2t(x):
	p, e = x
	return tuple(p.flatten()), tuple(e.flatten())

def t2x(t):
	p, e = t
	return np.asarray(p).reshape(2, -1), np.asarray(e).reshape(2, -1)

def extend_tree(G, p_rand):
	(V_prime, V_prime_goal, E_prime) = G
	x_nearest = get_nearest(V_prime, p_rand)
	p_nearest, e_nearest = x_nearest
	if _get_dist(p_nearest, p_goal) < R_goal:
		return V_prime, V_prime_goal, E_prime

	p_ref = steer(p_nearest, p_rand)
	X_set = get_primitive_set(x_nearest)
	x_new = get_new_vertex(p_ref, X_set)

	if is_obstacle_free(x_nearest, x_new):
		V_prime.add(x2t(x_new))
		E_prime.add((x2t(x_nearest), x2t(x_new)))

		if _get_dist(x_new[0], p_goal) < R_goal:
			V_prime_goal.add(x2t(x_new))

	return V_prime, V_prime_goal, E_prime

def _get_dist(p1, p2):
	return np.linalg.norm(p1 - p2)

def get_nearest(V, p_rand):
	# TODO: Convert to be compatible with RTrees for a massive speedup
	shortest = float("Inf")
	for t in V:
		p, e = t2x(t)
		d = _get_dist(p_rand, p)
		if d < shortest:
			shortest = d
			p_nearest = p
			e_nearest = e
	return (p_nearest, e_nearest)

def steer(p_nearest, p_rand):
	# dist = _get_dist(p_nearest, p_rand)
	# if dist > R_prim:
	# 	return p_nearest + (p_rand - p_nearest) * R_prim / dist
	return p_rand

def get_new_vertex(p_ref, X_set):
	# TODO: This might change with rtree implementation
	return get_nearest(X_set, p_ref)

def is_obstacle_free(x_1, x_2):
	# TODO: Obstacle checker
	curve = _get_bezier_curve(x_1, x_2)
	s_vals = np.linspace(0.0, 1.0, 30)
	samples = curve.evaluate_multi(s_vals)

	for o in obstacles:
		for s in samples:
			if _get_dist(s, np.asarray([o[0], o[1]])) < o[2]:
				return False
	return True

def get_sample(i):
	if np.random.rand() < bias_prob:
		return X_goal[0]
	return 1000 * (np.random.rand(2,1) - 0.5)

def get_cost(G, X_goal):
	J = set()
	V, V_goal, E = G
	for v in V_goal:
		X_goalarea = t2x(v)
		X_cost = get_vertex_cost(X_goalarea, X_goal)
		J.add((x2t(X_goalarea), X_cost))
	return sorted(J, key=lambda tup: tup[1])

def get_vertex_cost(X_goalarea, X_goal):
	p_goal, e_goal = X_goal
	p_goalarea, e_goalarea = X_goalarea

	e_to_goal = (p_goal - p_goalarea) / _get_dist(p_goal, p_goalarea)
	e_ref = 2*e_to_goal - e_goal

	cost_1 = c1 / (1 + e_to_goal.T.dot(e_goal).flatten())
	cost_2 = c2 / (1 + e_goalarea.T.dot(e_ref))
	cost_3 = c3 * tree_depth(X_goalarea) - (c1 + c2) / 2.
	return float(cost_1 + cost_2 + cost_3)


def tree_depth(x):
	# TODO: Implement this
	return 0.5

def get_results(A, G, J):
	V_prime, _, E_prime = G
	for j in J:
		X_goalarea = t2x(j[0])
		if is_obstacle_free(X_goalarea, X_goal):
			V_prime.add(x2t(X_goal))
			E_prime.add((x2t(X_goalarea), x2t(X_goal)))
			return cal_results(A, E_prime, T_arrival)
	raise Exception("No Path found.")

def get_paths(V, E, X_goalarea):
    V_list = list(V)
    V_dict = {k: v for v, k in enumerate(V_list)}
    graph = Graph(len(V_list))
    for (x1, x2) in E:
    	graph.add_edge(V_dict[x1], V_dict[x2])
    paths = graph.get_all_paths(V_dict[x2t(X_init)], V_dict[X_goalarea])
    return list(map(lambda path: list(map(lambda i: V_list[i], path)), paths))

# visualize_primitive_set()
main()


