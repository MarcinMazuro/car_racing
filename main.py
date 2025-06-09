import argparse
import os

import numpy as np
from gymnasium.wrappers import FrameStackObservation, ResizeObservation, GrayscaleObservation, NormalizeObservation
import gymnasium as gym
import datetime
import torch
from DQNAgent import DQNAgent
from PPOAgent import PPOAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUSTOM_ACTIONS = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Steer, Full Gas, Light Brake
    (-1, 1,   0), (0, 1,   0), (1, 1,   0),  # Steer, Full Gas, No Brake  (Action 4 is (0,1,0) - straight, gas)
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Steer, No Gas, Light Brake
    (-1, 0,   0), (0, 0,   0), (1, 0,   0)   # Steer, No Gas, No Brake
]

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, discrete_actions):
        super().__init__(env)
        self.discrete_actions = discrete_actions
        self.action_space = gym.spaces.Discrete(len(discrete_actions))

    def action(self, act):
        # Convert the tuple to a NumPy array with float32 dtype
        return np.array(self.discrete_actions[act], dtype=np.float32)

parser = argparse.ArgumentParser(description='Train or test agent for Car Racing')
parser.add_argument('--agent', type=str, choices=['dqn', 'ppo'],
                    help='Select agent: dqn or ppo')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                    help='Mode: train or test')
parser.add_argument('--render', action='store_true', default=False,
                    help='Render the environment')
parser.add_argument('--load_model', action='store_true', default=False,
                    help='Signal intent to load a model (requires --model_path for specification)')
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to a specific model file to load (for testing or resuming training)')
args = parser.parse_args()

print(f"Using device: {device}")

render_mode = "human" if args.render else "rgb_array"
env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)

if render_mode != "human":
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="videos/",
        # Use a simple episode counter from RecordVideo
        episode_trigger=lambda ep: ep % 5 == 0,  # Record 0, 5, 10, ...
        name_prefix="rl-video",
        disable_logger=True
    )
env = DiscreteActionWrapper(env, CUSTOM_ACTIONS)
image_size = 84
frame_stack_size = 4
env = ResizeObservation(env, (image_size, image_size))
env = GrayscaleObservation(env)
env = NormalizeObservation(env)
env = FrameStackObservation(env, frame_stack_size)

if args.agent == 'dqn':
    agent = DQNAgent(env, image_size, frame_stack_size)
else:
    agent = PPOAgent(env, image_size, frame_stack_size)


model_loaded_successfully = False

path_model_was_loaded_from = None

if args.model_path:
    if args.load_model:
        print(f"Attempting to load model from specified --model_path: {args.model_path}")
        try:
            agent.load(args.model_path)
            model_loaded_successfully = True
            path_model_was_loaded_from = args.model_path
            print(f"Model successfully loaded from {args.model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at --model_path: {args.model_path}")
            if args.mode == 'test':
                print("Exiting because model could not be loaded in test mode.")
                exit(1)
            print("Proceeding with a new model for training.")
        except Exception as e:
            print(f"An unexpected error occurred while loading from --model_path: {args.model_path}. Error: {e}")
            if args.mode == 'test':
                print("Exiting due to an unexpected error during model loading in test mode.")
                exit(1)
            print("Proceeding with a new model for training.")
    else:
        if args.mode == 'test':
            print(
                "Error: --model_path provided for test mode, but --load_model is false. A model must be loaded for testing.")
            print("Please use --load_model along with --model_path to test a specific model.")
            exit(1)

        print(
            f"Warning: --model_path '{args.model_path}' provided, but --load_model is false. This path will not be used for loading. New training run will save to a new unique path.")

elif args.load_model:  # --load_model is true, but --model_path is NOT given
    print("Error: --load_model was specified, but no --model_path was provided to identify which model to load.")
    if args.mode == 'test':
        print("For test mode, please provide the model path using --model_path.")
        exit(1)
    else:  # train mode
        print("For resuming training, please provide the model path using --model_path. Starting new training.")
        # model_loaded_successfully remains False

# --- Execution Logic ---
print(f"Env name: Car Racing")
print(f"Mode: {args.mode}")
print(f"Agent: {args.agent}")

if args.mode == 'train':
    current_run_save_dir_name = 'run-{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    training_checkpoint_save_dir = os.path.join('saved_models', 'car_racing', agent.name, current_run_save_dir_name)
    training_checkpoint_file_path = os.path.join(training_checkpoint_save_dir, 'model.pt')

    if model_loaded_successfully:
        print(f"Resuming training from model: {path_model_was_loaded_from}")
    else:
        print("Starting fresh training.")
    print(f"Checkpoints for this training session will be saved to: {training_checkpoint_file_path}")

    agent.train(checkpoint_save_path=training_checkpoint_file_path)
else:  # mode == 'test'
    if not model_loaded_successfully:
        print("Error: Test mode requires a model to be successfully loaded.")
        print("Please use --load_model and specify the model file with --model_path.")
        exit(1)
    # path_model_was_loaded_from should have been set if model_loaded_successfully is true
    print(f"Starting testing with model loaded from: {path_model_was_loaded_from}")
    agent.test(episodes=30)