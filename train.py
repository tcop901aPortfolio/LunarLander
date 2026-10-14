import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from my_lunar_lander import LunarLander

def make_env():
    env = LunarLander()                        
    env = TimeLimit(env, max_episode_steps=1000)  # end episode after 1000 frames
    return env


if __name__ == "__main__":
    # Run 8 games at once
    vec_env = make_vec_env(make_env, n_envs=8)

    model = PPO(
        "MlpPolicy",        # the neural network
        vec_env,
        n_steps=1024,       # frames each game plays before a learning update
        batch_size=64,
        n_epochs=4,
        gamma=0.999,        # reward factor
        gae_lambda=0.98,
        ent_coef=0.02,      # randomness
        verbose=1,          # print progress
    )

    model.learn(total_timesteps=5_000_000)
    model.save("my_lander_model")
    print("Saved to my_lander_model.zip")