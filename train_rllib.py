import ray
from ray.rllib.algorithms.ppo import PPOConfig
from environment import CustomOGEnv
from ray import train, tune
ray.init()

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment(CustomOGEnv)
    .env_runners(num_env_runners=1)
)

trainer = config.build()

for _ in range(1000):
    result = trainer.train()
    print(result)
