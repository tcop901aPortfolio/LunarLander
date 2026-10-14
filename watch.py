from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from my_lunar_lander import LunarLander
from scoring import score_episode

env = TimeLimit(LunarLander(render_mode="human"), max_episode_steps=1000)
model = PPO.load("./models/my_lander_modelv_4")

scores = []
for episode in range(5):
    obs, _ = env.reset()
    done, fires = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        fires += int(action != 0)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc

    landed = term and r == 100
    score = score_episode(landed, obs, fires)
    scores.append(score)
    result = "LANDED" if landed else ("CRASHED" if term else "TIMED OUT")
    print(f"Episode {episode + 1}: {result:9s} score {score:5.1f}%")

print(f"Average: {sum(scores) / len(scores):.1f}%")
env.close()