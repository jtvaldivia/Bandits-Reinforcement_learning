from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.RandomAgent import RandomAgent
from agents.IncrementalAgent import IncrementalAgent
from matplotlib import pyplot as plt


def show_results(bandit_results: type(dict)) -> None:
    plot_data = {}
    for epsilon, results in bandit_results.items():
        average_rewards = results.get_average_rewards()
        optimal_action_percentage = results.get_optimal_action_percentage()
        plot_data[epsilon] = [optimal_action_percentage , average_rewards]
    plt.plot(plot_data[0.1][0], label="Epsilon 0.1")
    plt.plot(plot_data[0.01][0], label="Epsilon 0.01")
    plt.plot(plot_data[0.0][0], label="Epsilon 0.0")
    plt.title("Optimal action percentage")
    plt.legend()
    plt.savefig("optimal_action_percentage.png")
    plt.show()
    plt.plot(plot_data[0.1][1], label="Epsilon 0.1")
    plt.plot(plot_data[0.01][1], label="Epsilon 0.01")
    plt.plot(plot_data[0.0][1], label="Epsilon 0.0")
    plt.title("Average rewards")
    plt.legend()
    plt.savefig("average_rewards.png")
    plt.show()

if __name__ == "__main__":
    epsilons = [0.1, 0.01, 0.0]
    all_epsilon_results = {}
    for epsilon in epsilons:
        
        NUM_OF_RUNS = 2000
        NUM_OF_STEPS = 1000

        results = BanditResults()
        for run_id in range(NUM_OF_RUNS):
            bandit = BanditEnv(seed=run_id)
            num_of_arms = bandit.action_space
            agent = IncrementalAgent(num_of_arms, epsilon)  # here you might change the agent that you want to use
            best_action = bandit.best_action
            for _ in range(NUM_OF_STEPS):
                action = agent.get_action()
                reward = bandit.step(action)
                agent.learn(action, reward)
                is_best_action = action == best_action
                results.add_result(reward, is_best_action)
            results.save_current_run()
        all_epsilon_results[epsilon] = results
        
    show_results(all_epsilon_results)
