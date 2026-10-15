import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load("best_model/best_model")        # load the trained brain

for attempt in range(5):
    obs, info = env.reset()
    done, total = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # AI picks the action
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    print(f"Attempt {attempt + 1}: score {total:.1f}")

env.close()