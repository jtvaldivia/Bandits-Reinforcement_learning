from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.GradientBanditAlgorithm import GradientBanditAlgorithm
from matplotlib import pyplot as plt


def show_results(bandit_results: type(dict)) -> None:
    plot_data = {}

    for alpha, results in bandit_results.items():
        for key, value in results.items():
            optimal_action_percentage = value.get_optimal_action_percentage()
            plot_data[key] = optimal_action_percentage
        
    # how can i show the alpha value
    plt.plot(plot_data[0], label="alpha = 0.4", color="lightblue")
    plt.plot(plot_data[1], label="alpha =0.4", color = "lightgreen")
    plt.plot(plot_data[2], label="alpha = 0.1", color="blue")
    plt.plot(plot_data[3], label="alpha = 0.1", color="green")
    plt.title("Optimal action percentage")
    plt.text(500, 0.75 , "with baseline")
    plt.text(500, 0.2, "without baseline ")
    plt.legend()
    plt.savefig("optimal_action_percentage.png")
    plt.show()


if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000
    alpha = {0.4, 0.1}
    results_mu = {}
    cantidad = 0
    results_total = {}
    for a in alpha:
        for i in [False, True]:
            mu = i
            results = BanditResults()
            for run_id in range(NUM_OF_RUNS):
                bandit = BanditEnv(seed=run_id, mean=4.0)
                num_of_arms = bandit.action_space
                agent = GradientBanditAlgorithm(num_of_arms, a, mu)  # here you might change the agent that you want to use
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
        results_total[a] = results_mu

    show_results(results_total)
