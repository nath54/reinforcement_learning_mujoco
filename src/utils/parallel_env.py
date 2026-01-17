"""
Vectorized environment running environments in parallel subprocesses
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

import numpy as np
from numpy.typing import NDArray

#
def worker(remote: Connection, parent_remote: Connection, env_fn_wrapper: Callable[[], Any]) -> None:
    """
    Worker process for parallel environment
    """

    # Close the parent remote
    parent_remote.close()

    # Try to create the environment
    try:
        env = env_fn_wrapper()
    #
    except Exception as e:
        # Print error message
        print(f"Worker initialization failed: {e}")
        # Print traceback
        traceback.print_exc()
        # Return
        return

    # Main loop for receiving commands
    while True:

        #
        try:

            # Receive command
            #
            cmd: str
            data: Any
            #
            cmd, data = remote.recv()

            # Handle different commands
            #
            ob: NDArray[np.float64]
            reward: float
            terminated: bool
            truncated: bool
            info: dict[str, Any]

            # Step command
            if cmd == 'step':

                # Take step in environment
                ob, reward, terminated, truncated, info = env.step(data)

                # If episode is done, reset environment
                if terminated or truncated:
                    #
                    res = env.reset()
                    if isinstance(res, tuple):
                        ob, _ = res
                    else:
                        ob = res

                # Send result to parent
                remote.send((ob, reward, terminated, truncated, info))

            # Reset command
            elif cmd == 'reset':

                # Reset environment
                res = env.reset()
                if isinstance(res, tuple):
                    ob, info = res
                else:
                    ob = res
                    info = {}


                # Send result to parent
                remote.send(ob)

            # Close command
            elif cmd == 'close':

                # Close remote
                remote.close()

                # Break loop
                break

            # Get spaces command
            elif cmd == 'get_spaces':

                # Send spaces to parent
                remote.send((env.observation_space, env.action_space))

            # Invalid command
            else:
                raise NotImplementedError

        #
        except Exception as e:

            #
            print(f"Worker error: {e}")
            #
            traceback.print_exc()

            # Break loop
            break


#
class SubprocVecEnv:
    """
    Vectorized environment running environments in parallel subprocesses
    """

    #
    def __init__(self, env_fns: list[Callable[[], Any]]) -> None:
        """
        Initialize the SubprocVecEnv
        """

        # Whether we are waiting for a step
        self.waiting: bool = False

        # Whether the environment is closed
        self.closed: bool = False

        # Number of environments
        nenvs: int = len(env_fns)

        # Create pipes for communication
        self.remotes: list[Connection]
        self.work_remotes: list[Connection]
        #
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        # Create worker processes
        self.ps: list[Process] = [
            Process(target=worker, args=(work_remote, remote, env_fn))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]

        # Start worker processes
        for p in self.ps:
            p.daemon = True
            p.start()

        # Close work remotes
        for remote in self.work_remotes:
            remote.close()

    #
    def step_async(self, actions: NDArray[Any] | list[Any]) -> None:
        """
        Send actions to the worker processes asynchronously
        """

        # Send step command to each environment
        #
        remote: Connection
        action: Any
        #
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        # Set waiting flag
        self.waiting = True

    #
    def step_wait(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], tuple[dict[str, Any], ...]]:
        """
        Wait for the results from the worker processes
        """

        # Receive results from each environment
        results: list[tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]] = [remote.recv() for remote in self.remotes]

        # Set waiting flag
        self.waiting = False

        # Unzip results
        obs, rews, terms, truncs, infos = zip(*results)

        # Stack results
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truncs), infos

    #
    def step(self, actions: NDArray[Any] | list[Any]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], tuple[dict[str, Any], ...]]:
        """
        Step the environment with the given actions
        """

        # Send step command to each environment
        self.step_async(actions)

        # Wait for results
        return self.step_wait()

    #
    def reset(self) -> NDArray[np.float64]:
        """
        Reset the environments
        """

        # Send reset command to each environment
        for remote in self.remotes:
            remote.send(('reset', None))

        # Receive results from each environment
        return np.stack([remote.recv() for remote in self.remotes])

    #
    def close(self) -> None:
        """
        Close the environments
        """

        # If already closed, return
        if self.closed:
            return

        # If waiting for step, wait for results
        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        # Send close command to each environment
        for remote in self.remotes:
            remote.send(('close', None))

        # Wait for processes to finish
        for p in self.ps:
            p.join()

        # Set closed flag
        self.closed = True
