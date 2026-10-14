from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from my_lunar_lander import LunarLander

env = TimeLimit(LunarLander(render_mode="human"), max_episode_steps=1000)
model = PPO.load("./models/my_lander_modelv_1")

for episode in range(5):
    obs, _ = env.reset()
    done = False
    total = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    print(f"Episode {episode + 1}: score {total:.1f}")
env.close()