"""Test DQN performance on cartpole."""
import pytest
from torch.nn import functional as F  # NOQA

from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch.algos import DQN
from garage.torch.policies import DiscreteQFArgmaxPolicy
from garage.torch.q_functions import DiscreteMLPQFunction
from garage.trainer import Trainer

from tests.fixtures import snapshot_config


@pytest.mark.large
def test_dqn_cartpole():
    set_seed(24)
    trainer = Trainer(snapshot_config)

    n_epochs = 11
    steps_per_epoch = 10
    sampler_batch_size = 512
    num_timesteps = 100 * steps_per_epoch * sampler_batch_size
    env = GymEnv('CartPole-v0')

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(8, 5))

    policy = DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)
    exploration_policy = EpsilonGreedyPolicy(env_spec=env.spec,
                                             policy=policy,
                                             total_timesteps=num_timesteps,
                                             max_epsilon=1.0,
                                             min_epsilon=0.01,
                                             decay_ratio=0.4)
    algo = DQN(env_spec=env.spec,
               policy=policy,
               qf=qf,
               exploration_policy=exploration_policy,
               replay_buffer=replay_buffer,
               steps_per_epoch=steps_per_epoch,
               qf_lr=5e-5,
               discount=0.9,
               min_buffer_size=int(1e4),
               n_train_steps=500,
               target_update_freq=30,
               buffer_batch_size=64)

    trainer.setup(algo, env, sampler_cls=LocalSampler)
    last_avg_return = trainer.train(n_epochs=n_epochs,
                                    batch_size=sampler_batch_size)
    assert last_avg_return > 10
    env.close()
