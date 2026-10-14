import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from my_lunar_lander import LunarLander
from scoring import score_episode

def evaluate(model_path, episodes=100):
    env = TimeLimit(LunarLander(), max_episode_steps=1000)
    model = PPO.load(model_path)
    scores = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done, fires = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            fires += int(action != 0)                # any engine firing
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
        landed = term and r == 100
        scores.append(score_episode(landed, obs, fires))

    print(f"{model_path}: {np.mean(scores):.1f}% "
          f"(best {np.max(scores):.0f}, worst {np.min(scores):.0f})")

if __name__ == "__main__":
    evaluate("./models/my_lander_modelv_1")