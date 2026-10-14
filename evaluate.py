import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from my_lunar_lander import LunarLander

def evaluate(model_path, episodes=100):
    env = TimeLimit(LunarLander(), max_episode_steps=1000)
    model = PPO.load(model_path)
    scores, landed = [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done, total = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        scores.append(total)
        landed += total >= 200
    print(f"{model_path}: mean {np.mean(scores):.1f} | std {np.std(scores):.1f} "
          f"| best {np.max(scores):.1f} | worst {np.min(scores):.1f} "
          f"| solved {landed}/{episodes}")

if __name__ == "__main__":
    evaluate("./models/my_lander_modelv_3")