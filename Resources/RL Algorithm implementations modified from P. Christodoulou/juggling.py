import gym
from agents.hierarchical_agents.DIAYN import DIAYN
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.SAC import SAC
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config

config = Config()
#config.environment = [gym.make('Bowling-ram-v0'), gym.make('Pong-ram-v0'), gym.make('SpaceInvaders-ram-v0')]
config.environment_name = 'MountainCarContinuous-v0'
config.environment = gym.make(config.environment_name)
config.seed = 1
config.env_parameters = {}
config.num_episodes_to_run = 11
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 3
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False

linear_hidden_units = [128, 128, 32]
learning_rate = 0.01
buffer_size = 100000
batch_size = 256
batch_norm = False
embedding_dimensionality = 10
gradient_clipping_norm = 5
update_every_n_steps = 1
learning_iterations = 2
epsilon_decay_rate_denominator = 400
discount_rate = 0.99
tau = 0.01
sequitur_k = 2
pre_training_learning_iterations_multiplier = 50
episodes_to_run_with_no_exploration = 10
action_balanced_replay_buffer = True
copy_over_hidden_layers = True
action_length_reward_bonus = 0.1

num_skills = 30
num_unsupservised_episodes = int(.75 * config.num_episodes_to_run)
discriminator_learning_rate = 0.0003
timesteps_to_give_up_control_for = 30
config.hyperparameters = {

    "DIAYN": {
        "DISCRIMINATOR": {
            "final_layer_activation": None,
            "learning_rate": discriminator_learning_rate,
            "linear_hidden_units": linear_hidden_units,
            "gradient_clipping_norm": 5,
        },
        "AGENT": {
            "steps_per_env": timesteps_to_give_up_control_for,
            "clip_rewards": False,
            "do_evaluation_iterations": False,
            "learning_rate": 0.005,
            "linear_hidden_units": [128, 128, 32],
            "final_layer_activation": ["SOFTMAX", None],
            "gradient_clipping_norm": 5.0,
            "epsilon_decay_rate_denominator": 1.0,
            "normalise_rewards": True,
            "exploration_worker_difference": 2.0,
            "min_steps_before_learning": 10000,
            "batch_size": 256,
            "discount_rate": 0.99,
            # questionable...
            "mu": 0.0,  # for O-H noise
            "theta": 0.15,  # for O-H noise
            "sigma": 0.25,  # for O-H noise
            "update_every_n_steps": 1,
            "learning_updates_per_learning_session": 1,
            "automatically_tune_entropy_hyperparameter": True,
            "entropy_term_weight": None,
            "add_extra_noise": False,
            "use_GPU": config.use_GPU,
            "Actor": {
                "learning_rate": 0.0003,
                "linear_hidden_units": [128, 128, 32],
                "final_layer_activation": None,
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier",
            },

            "Critic": {
                "learning_rate": 0.0003,
                "linear_hidden_units": [128, 128, 32],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier",
            },
        },
        "MANAGER": {
            "timesteps_to_give_up_control_for": timesteps_to_give_up_control_for,
            "learning_rate": 0.01,
            "batch_size": 256,
            "buffer_size": 40000,
            "epsilon": 1.0,
            "epsilon_decay_rate_denominator": 1,
            "discount_rate": 0.99,
            "tau": 0.01,
            "alpha_prioritised_replay": 0.6,
            "beta_prioritised_replay": 0.1,
            "incremental_td_error": 1e-8,
            "update_every_n_steps": 1,
            "linear_hidden_units": [30, 15],
            "final_layer_activation": "None",
            "batch_norm": False,
            "gradient_clipping_norm": 0.7,
            "learning_iterations": 1,
            "clip_rewards": False
        },

        "num_skills": num_skills,
        "num_unsupservised_episodes": num_unsupservised_episodes,
        "final_layer_activation": None
    },
    "Actor_Critic_Agents": {
        'batch_size': 256,
        "clip_rewards": False,
        'automatically_tune_entropy_hyperparameter': True,
        'entropy_term_weight': .3,
        'add_extra_noise': False,
        'learning_updates_per_learning_session': 1,
        'min_steps_before_learning': 10000,
        'update_every_n_steps': 1,
        'discount_rate': .99,
        'do_evaluation_iterations': False,
        "Actor": {
            "learning_rate": 0.0003,
                "linear_hidden_units": [128, 128, 32],
                "final_layer_activation": None,
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier",
        },

        "Critic": {
            "learning_rate": 0.0003,
                "linear_hidden_units": [128, 128, 32],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier",
        },
    }
}

if __name__ == "__main__":
    AGENTS = [SAC]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
