#include <vector>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const int N_SAMPLE = 10;
const float PROB_TO_GOAL = 1.;

class SplineRRT {
	Vector2f goal;
	float search_space[];

	Vector2f get_random() {
		Vector2f minimum;
		minimum << search_space[0], search_space[2];
		Vector2f bounds;
		bounds << search_space[1] - search_space[0], search_space[3] - search_space[2];
		Vector2f rand = Vector2f::Random();
		return minimum + (bounds + bounds.cwiseProduct(rand)) / 2;
	}

	Vector2f get_sample() {
		float r = (float) rand() / RAND_MAX;
		if (r < PROB_TO_GOAL)
			return goal;
		return get_random();
	}

	public:
		SplineRRT(float search_space[4]) {
			search_space = search_space;
		}

		int run() {
			goal = get_random();
			Vector2f p_rand;

			// Generate the random graph
			for (int i=0; i < N_SAMPLE; i++) {
				p_rand = get_sample();
				cout << p_rand << endl << endl;
			}
			return 0;
		}
};

int main() {
	// Random seed
	srand((unsigned int) time(0));

	// MIN_X, MAX_X, MIN_Y, MAX_Y
	float search_space[4] = {100, 200, 100, 200};
	
	SplineRRT rrt(search_space);
	rrt.run();
	
}