from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.GradientBanditAlgorithm import GradientBanditAlgorithm
from matplotlib import pyplot as plt


def show_results(bandit_results: type(dict)) -> None:
    plot_data = {}
    for mu, results in bandit_results.items():
        optimal_action_percentage = results.get_optimal_action_percentage()
        plot_data[mu] = optimal_action_percentage
        

    plt.plot(plot_data[0], label="Mu 0.0")
    plt.plot(plot_data[1], label="Mu 4.0")
    plt.title("Optimal action percentage")
    plt.savefig("optimal_action_percentage.png")
    plt.show()


if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000
    alpha = 0.4
    results_mu = {}
    cantidad = 0
    for i in [False, True]:
        mu = i
        results = BanditResults()
        for run_id in range(NUM_OF_RUNS):
            bandit = BanditEnv(seed=run_id, mean=4.0)
            num_of_arms = bandit.action_space
            agent = GradientBanditAlgorithm(num_of_arms, alpha, mu)  # here you might change the agent that you want to use
            best_action = bandit.best_action
            for _ in range(NUM_OF_STEPS):
                action = agent.get_action()
                reward = bandit.step(action)
                agent.learn(action, reward)
                is_best_action = action == best_action
                results.add_result(reward, is_best_action)
            results.save_current_run()
        results_mu[cantidad] = results
        cantidad += 1

    show_results(results_mu)
