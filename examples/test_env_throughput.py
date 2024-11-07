import time
from typing import List, Tuple, Union

import torch
import tqdm
from torchrl._utils import timeit
from torchrl.collectors import (MultiaSyncDataCollector,
                                MultiSyncDataCollector, SyncDataCollector)
from torchrl.envs import EnvCreator, GymEnv, ParallelEnv
from torchrl.envs.libs.gym import gym_backend as gym_bc
from torchrl.envs.libs.gym import set_gym_backend
from torchrl.envs.utils import RandomPolicy


def test_pure_gym(
    envname: str,
    num_workers: int,
    gym_backend: str,
    total_frames: int = None,
    log: tqdm.tqdm = None,
) -> None:
    """Test the performance of pure Gym's AsyncVectorEnv.

    Args:
        envname (str): Name of the environment.
        num_workers (int): Number of worker processes.
        gym_backend (str): Gym backend to use ('gym' or 'gymnasium').
        total_frames (int): Total number of frames to process.
        log (tqdm.tqdm): Log file to write results.
    """

    def make(envname: str, gym_backend: str) -> gym_bc:
        with set_gym_backend(gym_backend):
            return gym_bc().make(envname)

    with set_gym_backend(gym_backend):
        env = gym_bc().vector.AsyncVectorEnv([
            make(envname=envname, gym_backend=gym_backend)
            for _ in range(num_workers)
        ])
    env.reset()
    global_step = 0
    start = time.time()
    for _ in tqdm.tqdm(range(total_frames // num_workers)):
        env.step(env.action_space.sample())
        global_step += num_workers
    env.close()
    log.write(f'pure gym: {total_frames/ (time.time() - start): 4.4f} fps\n')
    log.flush()


def test_parallel_env(
    envname: str,
    num_workers: int,
    gym_backend: str,
    total_frames: int,
    device: Union[str, torch.device],
    log: tqdm.tqdm,
) -> None:
    """Test the performance of TorchRL's ParallelEnv with regular GymEnv
    instances.

    Args:
        envname (str): Name of the environment.
        num_workers (int): Number of worker processes.
        gym_backend (str): Gym backend to use ('gym' or 'gymnasium').
        total_frames (int): Total number of frames to process.
        device (Union[str, torch.device]): Device to use ('cpu' or 'cuda:0').
        log (tqdm.tqdm): Log file to write results.
    """

    def make(envname: str, gym_backend: str, device: str) -> GymEnv:
        with set_gym_backend(gym_backend):
            return GymEnv(envname, device=device)

    penv = ParallelEnv(
        num_workers,
        EnvCreator(make(envname, gym_backend, device)),
        device=device,
    )

    with torch.inference_mode():
        penv.rollout(2)
        pbar = tqdm.tqdm(total=total_frames)
        t0 = time.time()
        data = None
        for _ in range(100):
            data = penv.rollout(100, break_when_any_done=False, out=data)
            pbar.update(100 * num_workers)
    log.write(
        f'penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n')
    log.flush()
    penv.close()
    timeit.print()
    del penv


def test_single_collector_with_penv(
    envname: str,
    num_workers: int,
    gym_backend: str,
    total_frames: int,
    device: Union[str, torch.device],
    log: tqdm.tqdm,
) -> None:
    """Test the performance of a single SyncDataCollector with TorchRL's
    ParallelEnv.

    Args:
        envname (str): Name of the environment.
        num_workers (int): Number of worker processes.
        gym_backend (str): Gym backend to use ('gym' or 'gymnasium').
        total_frames (int): Total number of frames to process.
        device (Union[str, torch.device]): Device to use ('cpu' or 'cuda:0').
        log (tqdm.tqdm): Log file to write results.
    """

    def make(envname: str, gym_backend: str, device: str) -> GymEnv:
        with set_gym_backend(gym_backend):
            return GymEnv(envname, device=device)

    env_make = EnvCreator(make(envname, gym_backend, device=device))
    penv = ParallelEnv(num_workers, env_make, device=device)
    collector = SyncDataCollector(
        penv,
        RandomPolicy(penv.action_spec),
        frames_per_batch=1024,
        total_frames=total_frames,
        device=device,
    )
    pbar = tqdm.tqdm(total=total_frames)
    t0 = time.time()
    frames = 0
    for data in collector:
        frames += data.numel()
        pbar.update(data.numel())
        pbar.set_description(
            f'single collector + torchrl penv: {frames / (time.time() - t0): 4.4f} fps'
        )
    log.write(
        f'single collector + torchrl penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n'
    )
    log.flush()
    collector.shutdown()
    del collector


def test_gym_penv(
    envname: str,
    num_workers: int,
    gym_backend: str,
    total_frames: int,
    device: Union[str, torch.device],
    log: tqdm.tqdm,
) -> None:
    """Test the performance of Gym's ParallelEnv.

    Args:
        envname (str): Name of the environment.
        num_workers (int): Number of worker processes.
        gym_backend (str): Gym backend to use ('gym' or 'gymnasium').
        total_frames (int): Total number of frames to process.
        device (Union[str, torch.device]): Device to use ('cpu' or 'cuda:0').
        log (tqdm.tqdm): Log file to write results.
    """

    def make_env(
        envname: str,
        num_workers: int,
        gym_backend: str,
        device: Union[str, torch.device],
    ) -> GymEnv:
        with set_gym_backend(gym_backend):
            penv = GymEnv(envname, num_envs=num_workers, device=device)
        return penv

    penv = make_env(envname, num_workers, gym_backend, device)
    penv.rollout(2)
    pbar = tqdm.tqdm(total=total_frames)
    t0 = time.time()
    for _ in range(100):
        penv.rollout(100, break_when_any_done=False)
        pbar.update(num_workers * 100)
    log.write(
        f'gym penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n')
    log.flush()
    penv.close()
    del penv


def test_async_collector_with_penv(
    envname: str,
    num_workers: int,
    num_collectors: int,
    gym_backend: str,
    total_frames: int,
    device: Union[str, torch.device],
    log: tqdm.tqdm,
) -> None:
    """Test the performance of MultiaSyncDataCollector with TorchRL's
    ParallelEnv.

    Args:
        envname (str): Name of the environment.
        num_workers (int): Number of worker processes.
        num_collectors (int): Number of data collectors.
        gym_backend (str): Gym backend to use ('gym' or 'gymnasium').
        total_frames (int): Total number of frames to process.
        device (Union[str, torch.device]): Device to use ('cpu' or 'cuda:0').
        log (tqdm.tqdm): Log file to write results.
    """

    def make_env(envname: str, gym_backend: str, device: str) -> GymEnv:
        with set_gym_backend(gym_backend):
            return GymEnv(envname, device='cpu')

    penv = ParallelEnv(
        num_workers // num_collectors,
        EnvCreator(make_env(
            envname,
            gym_backend,
            device,
        )),
        device=device,
    )
    collector = MultiaSyncDataCollector(
        [penv] * num_collectors,
        policy=RandomPolicy(penv.action_spec),
        frames_per_batch=1024,
        total_frames=total_frames,
        device=device,
    )
    pbar = tqdm.tqdm(total=total_frames)
    frames = 0
    for i, data in enumerate(collector):
        if i == num_collectors:
            t0 = time.time()
        if i >= num_collectors:
            frames += data.numel()
            pbar.update(data.numel())
            pbar.set_description(
                f'collector + torchrl penv: {frames / (time.time() - t0): 4.4f} fps'
            )
    log.write(
        f'async collector + torchrl penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n'
    )
    log.flush()
    collector.shutdown()
    del collector


def test_async_collector_with_gym_penv(
    envname: str,
    num_workers: int,
    num_collectors: int,
    gym_backend: str,
    total_frames: int,
    device: Union[str, torch.device],
    log: tqdm.tqdm,
) -> None:
    """Test the performance of MultiaSyncDataCollector with Gym's ParallelEnv.

    Args:
        envname (str): Name of the environment.
        num_workers (int): Number of worker processes.
        num_collectors (int): Number of data collectors.
        gym_backend (str): Gym backend to use ('gym' or 'gymnasium').
        total_frames (int): Total number of frames to process.
        device (Union[str, torch.device]): Device to use ('cpu' or 'cuda:0').
        log (tqdm.tqdm): Log file to write results.
    """

    def make_env(envname: str, num_workers: int, gym_backend: str) -> GymEnv:
        with set_gym_backend(gym_backend):
            penv = GymEnv(envname, num_envs=num_workers, device=device)
        return penv

    penv = EnvCreator(
        lambda num_workers=num_workers // num_collectors: make_env(
            envname=envname, num_workers=num_workers, gym_backend=gym_backend))
    collector = MultiaSyncDataCollector(
        [penv] * num_collectors,
        policy=RandomPolicy(penv().action_spec),
        frames_per_batch=1024,
        total_frames=total_frames,
        num_sub_threads=num_workers // num_collectors,
        device=device,
    )
    pbar = tqdm.tqdm(total=total_frames)
    frames = 0
    for i, data in enumerate(collector):
        if i == num_collectors:
            t0 = time.time()
        if i >= num_collectors:
            frames += data.numel()
            pbar.update(data.numel())
            pbar.set_description(
                f'{i} collector + gym penv: {frames / (time.time() - t0): 4.4f} fps'
            )
    log.write(
        f'async collector + gym penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n'
    )
    log.flush()
    collector.shutdown()
    del collector


def test_sync_collector_with_penv(
    envname: str,
    num_workers: int,
    num_collectors: int,
    gym_backend: str,
    total_frames: int,
    device: Union[str, torch.device],
    log: tqdm.tqdm,
) -> None:
    """Test the performance of MultiSyncDataCollector with TorchRL's
    ParallelEnv.

    Args:
        envname (str): Name of the environment.
        num_workers (int): Number of worker processes.
        num_collectors (int): Number of data collectors.
        gym_backend (str): Gym backend to use ('gym' or 'gymnasium').
        total_frames (int): Total number of frames to process.
        device (Union[str, torch.device]): Device to use ('cpu' or 'cuda:0').
        log (tqdm.tqdm): Log file to write results.
    """

    def make_env(envname: str, gym_backend: str) -> GymEnv:
        with set_gym_backend(gym_backend):
            return GymEnv(envname, device=device)

    penv = ParallelEnv(
        num_workers // num_collectors,
        EnvCreator(make_env(envname, gym_backend)),
        device=device,
    )
    collector = MultiSyncDataCollector(
        [penv] * num_collectors,
        policy=RandomPolicy(penv.action_spec),
        frames_per_batch=1024,
        total_frames=total_frames,
        device=device,
    )
    pbar = tqdm.tqdm(total=total_frames)
    frames = 0
    for i, data in enumerate(collector):
        if i == num_collectors:
            t0 = time.time()
        if i >= num_collectors:
            frames += data.numel()
            pbar.update(data.numel())
            pbar.set_description(
                f'collector + torchrl penv: {frames / (time.time() - t0): 4.4f} fps'
            )
    log.write(
        f'sync collector + torchrl penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n'
    )
    log.flush()
    collector.shutdown()
    del collector


def test_sync_collector_with_gym_penv(
    envname: str,
    num_workers: int,
    num_collectors: int,
    gym_backend: str,
    total_frames: int,
    device: Union[str, torch.device],
    log: tqdm.tqdm,
) -> None:
    """Test the performance of MultiSyncDataCollector with Gym's ParallelEnv.

    Args:
        envname (str): Name of the environment.
        num_workers (int): Number of worker processes.
        num_collectors (int): Number of data collectors.
        gym_backend (str): Gym backend to use ('gym' or 'gymnasium').
        total_frames (int): Total number of frames to process.
        device (Union[str, torch.device]): Device to use ('cpu' or 'cuda:0').
        log (tqdm.tqdm): Log file to write results.
    """

    def make_env(envname: str, num_workers: int, gym_backend: str) -> GymEnv:
        with set_gym_backend(gym_backend):
            penv = GymEnv(envname, num_envs=num_workers, device=device)
        return penv

    penv = EnvCreator(lambda num_workers=num_workers // num_collectors:
                      make_env(envname, num_workers, gym_backend))
    collector = MultiSyncDataCollector(
        [penv] * num_collectors,
        policy=RandomPolicy(penv().action_spec),
        frames_per_batch=1024,
        total_frames=total_frames,
        num_sub_threads=num_workers // num_collectors,
        device=device,
    )
    pbar = tqdm.tqdm(total=total_frames)
    frames = 0
    for i, data in enumerate(collector):
        if i == num_collectors:
            t0 = time.time()
        if i >= num_collectors:
            frames += data.numel()
            pbar.update(data.numel())
            pbar.set_description(
                f'{i} collector + gym penv: {frames / (time.time() - t0): 4.4f} fps'
            )
    log.write(
        f'sync collector + gym penv {device}: {total_frames / (time.time() - t0): 4.4f} fps\n'
    )
    log.flush()
    collector.shutdown()
    del collector


def main() -> None:
    """Main function to run the performance tests for various
    configurations."""
    avail_devices: Tuple[Union[str, torch.device], ...] = ('cpu', )
    if torch.cuda.device_count():
        avail_devices = avail_devices + ('cuda:0', )

    env_names: List[str] = [
        'CartPole-v1',
        'Pendulum-v1',
        'HalfCheetah-v4',
        'myoHandReachRandom-v0',
        'ALE/Breakout-v5',
    ]

    num_workers_collectors: List[Tuple[int, int]] = [(32, 8), (64, 8), (8, 2),
                                                     (16, 4)]

    for envname in env_names:
        for num_workers, num_collectors in num_workers_collectors:
            log_filename: str = f'{envname}_{num_workers}.txt'.replace(
                '/', '-')
            with open(log_filename, 'w+') as log:
                if 'myo' in envname:
                    gym_backend: str = 'gym'
                else:
                    gym_backend = 'gymnasium'

                total_frames: int = num_workers * 10_1000

                test_pure_gym(envname, num_workers, gym_backend, total_frames,
                              log)

                for device in avail_devices:
                    test_parallel_env(envname, num_workers, gym_backend,
                                      total_frames, device, log)

                for device in avail_devices:
                    test_single_collector_with_penv(envname, num_workers,
                                                    gym_backend, total_frames,
                                                    device, log)

                for device in avail_devices:
                    test_gym_penv(envname, num_workers, gym_backend,
                                  total_frames, device, log)

                for device in avail_devices:
                    test_async_collector_with_penv(
                        envname,
                        num_workers,
                        num_collectors,
                        gym_backend,
                        total_frames,
                        device,
                        log,
                    )

                for device in avail_devices:
                    test_async_collector_with_gym_penv(
                        envname,
                        num_workers,
                        num_collectors,
                        gym_backend,
                        total_frames,
                        device,
                        log,
                    )

                for device in avail_devices:
                    test_sync_collector_with_penv(
                        envname,
                        num_workers,
                        num_collectors,
                        gym_backend,
                        total_frames,
                        device,
                        log,
                    )

                for device in avail_devices:
                    test_sync_collector_with_gym_penv(
                        envname,
                        num_workers,
                        num_collectors,
                        gym_backend,
                        total_frames,
                        device,
                        log,
                    )


if __name__ == '__main__':
    main()
