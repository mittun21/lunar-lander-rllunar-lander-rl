# 🚀 Lunar Lander: Teaching an AI to Land a Rocket

An AI that teaches itself to land a rocket on the Moon by trial and error,
using reinforcement learning (PPO) in Gymnasium's `LunarLander-v3`.

## What this project does

At the start, the AI presses random buttons and crashes every time.
After practicing for about a million frames, it learns to fly down gently
and land between the flags. Nobody tells it how to land: it only gets points
for good moves and loses points for crashing.

## How it works

- **Environment:** `LunarLander-v3` from Gymnasium (Box2D physics)
- **What the AI sees:** 8 numbers (position, speed, tilt, spin, legs touching). It never sees the picture
- **What the AI can do:** 4 actions: nothing, left engine, main engine, right engine
- **Brain:** a small neural network (2 layers of 64), `MlpPolicy`
- **Learning method:** PPO from Stable-Baselines3
- **Reward:** closer, slower and upright earns points; fuel costs points; +100 for landing, -100 for crashing

## Files

| File | What it does |
|---|---|
| `crash.py` | Random actions: watch the rocket crash before any training |
| `train.py` | Trains the AI and saves its brain |
| `watch.py` | Loads the trained brain and flies 5 landings |
| `ppo_lander.zip` | Trained brain from my first run |
| `best_model/best_model.zip` | Best brain from my faster run |

## Results

| Run | Setup | Training time | Score |
|---|---|---|---|
| 1 | 16 rockets, batch size 64 | [X] min | [X] |
| 2 | 16 rockets, batch size 512, 1 PyTorch thread, stops at score 200 | [X] min | [X] |

A score of 200+ counts as "solved."

**Speeding it up:** I noticed training ran at about 12,700 frames/sec while the
rockets were only flying, but dropped to about 1,900 once the AI started updating
its brain. So the bottleneck was the learning step, not the game. Bigger batches,
a single PyTorch thread and early stopping cut training time from [X] to [X] minutes.

## Run it yourself (Windows)

    python -m venv venv
    venv\Scripts\python -m pip install -r requirements.txt
    venv\Scripts\python crash.py
    venv\Scripts\python train.py
    venv\Scripts\python watch.py

## What's next

- [ ] Custom reward rules with a Gymnasium wrapper
- [ ] Reward hacking: reward the AI for hovering, watch it refuse to land, then fix it
- [ ] Hard mode: stronger gravity, wind and turbulence
- [ ] Record a video of the landing

## Credits

Built with [Gymnasium](https://gymnasium.farama.org) and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io).
