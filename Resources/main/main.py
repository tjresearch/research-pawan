import gym, os
from agents.hierarchical_agents.DIAYN import DIAYN
from agents.hierarchical_agents.DBH import DBH
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.SAC import SAC
from agents.DQN_agents.DDQN import DDQN
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
import argparse


config = Config()
parser = argparse.ArgumentParser()
parser.add_argument('--env', action='store', dest='environment', default='SpaceInvaders-v0', help='which environment to compare on')
parser.add_argument('--alg', nargs='+', action='store', dest='algorithms', default='SAC_Discrete', help='which algorithms to compare')
parser.add_argument('--eval', type=bool, default=False, action='store', dest='evaluate',
                    help='set False for training and True for evaluating.')
parser.add_argument('--num_ep', type=int, default=20, action='store', dest='num_episodes',
                    help='How many episodes to train for')
parser.add_argument('--save_results', type=bool, default=True, action='store', dest='save_results',
                    help='Set to False if you don\'t want to save training results and the model')
parser.add_argument('--run_triathlon_standard', type=bool, default=False, action='store', dest='rts',
                    help='Run Triathlon with SAC_Discrete, DDQN, DIAYN, and DBH, with standard configurations')
parser.add_argument('--n_trials', type=int, default='1', action='store', dest='n_trials',
                    help='How many training runs to do per agent to analyze variance')
parser.add_argument('--seed', type=int, default='1', action='store', dest='seed',
                    help='Set the seed to reproduce results')
parser.add_argument('--use_GPU', type=bool, default=True, action='store', dest='use_GPU',
                    help='Set to False if you don\'t have a GPU')
parser.add_argument('--run_prefix', action='store', dest='run_prefix', default='run_1',
                    help='Add a prefix to this run to group results together. Runs will be saved in results/run_prefix'
                         'allows user to eval or train existing models by specifying this run')
parser.add_argument('--train_existing_model', action='store', dest='tem', default=False,
                    help='If you want to continue training an existing model, setting this to true will find the model '
                         'associated with the run prefix, environment and algorithm specified')
args = parser.parse_args()

str_to_obj = {
    'SAC': SAC,
    'DDQN': DDQN,
    'SAC_Discrete': SAC_Discrete,
    'DIAYN': DIAYN,
    'DBH': DBH
}
if args.rts:
    config.rts()
    AGENTS = [DDQN, SAC_Discrete, DIAYN, DBH]

else:
    AGENTS = [str_to_obj[i] for i in args.algorithms]
    config.environment_name = args.environment
    config.environment = gym.make(config.environment_name)
    config.eval = args.evaluate
    config.seed = args.seed
    config.num_episodes_to_run = args.num_episodes
    config.runs_per_agent = args.n_trials
    config.use_GPU = args.use_GPU
    config.save_results = args.save_results
    config.run_prefix = args.run_prefix
    config.save_directory = 'results/{}'.format(config.run_prefix)
    if not os.path.exists(config.save_directory):
        os.makedirs(config.save_directory)
    config.visualise_overall_agent_results = True
    config.standard_deviation_results = 1.0


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
    print('rerun with -h flag to see possible args or check the read me file')

    trainer = Trainer(config, AGENTS)
    if config.eval:
        trainer.eval_model(config.num_episodes_to_run)
    else:
        trainer.run_games_for_agents()

