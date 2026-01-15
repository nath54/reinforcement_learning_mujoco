from multiprocessing import Process, Pipe
from typing import Any, List, Tuple
import numpy as np

def worker(remote: Any, parent_remote: Any, env_fn_wrapper: Any) -> None:
    """Worker process for parallel environment"""
    parent_remote.close()

    try:
        env = env_fn_wrapper()
    except Exception as e:
        print(f"Worker initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                ob, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    ob, _ = env.reset()
                remote.send((ob, reward, terminated, truncated, info))

            elif cmd == 'reset':
                ob, info = env.reset()
                remote.send(ob)

            elif cmd == 'close':
                remote.close()
                break

            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))

            else:
                raise NotImplementedError

        except Exception as e:
            print(f"Worker error: {e}")
            import traceback
            traceback.print_exc()
            break

class SubprocVecEnv:
    """Vectorized environment running environments in parallel subprocesses"""

    def __init__(self, env_fns: List[Any]) -> None:
        self.waiting: bool = False
        self.closed: bool = False
        nenvs: int = len(env_fns)

        self.remotes: List[Any]
        self.work_remotes: List[Any]
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps: List[Process] = [
            Process(target=worker, args=(work_remote, remote, env_fn))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions: Any) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self) -> Tuple[Any, Any, Any, Any, Any]:
        results: List[Any] = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, terms, truncs, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truncs), infos

    def step(self, actions: Any) -> Tuple[Any, Any, Any, Any, Any]:
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> Any:
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self) -> None:
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(('close', None))

        for p in self.ps:
            p.join()

        self.closed = True