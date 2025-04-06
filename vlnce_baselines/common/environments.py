from typing import Any, Dict, Optional, Tuple, List, Union

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

    def get_metrics(self):
        return self.habitat_env.get_metrics()

    def get_geodesic_dist(self, 
        node_a: List[float], node_b: List[float]):
        return self._env.sim.geodesic_distance(node_a, node_b)

    def check_navigability(self, node: List[float]):
        return self._env.sim.is_navigable(node)

    def get_agent_info(self):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }

    def get_observation_at(self,
        source_position: List[float],
        source_rotation: List[Union[int, np.float64]],
        keep_agent_at_new_pose: bool = False):
        return self._env.sim.get_observations_at(
            source_position,
            source_rotation,
            keep_agent_at_new_pose)

    def observations_by_angles(self, angle_list: List[float]):
        r'''for getting observations from desired angles
        requires rad, positive represents anticlockwise'''
        obs = []
        sim = self._env.sim
        init_state = sim.get_agent_state()
        prev_angle = 0
        left_action = HabitatSimActions.TURN_LEFT
        init_amount = sim.get_agent(0).agent_config.action_space[left_action].actuation.amount # turn left
        for angle in angle_list:
            sim.get_agent(0).agent_config.action_space[left_action].actuation.amount = (angle-prev_angle)*180/np.pi
            obs.append(sim.step(left_action))
            prev_angle = angle
        sim.set_agent_state(init_state.position, init_state.rotation)
        sim.get_agent(0).agent_config.action_space[left_action].actuation.amount = init_amount
        return obs

    def current_dist_to_goal(self):
        sim = self._env.sim
        init_state = sim.get_agent_state()
        init_distance = self._env.sim.geodesic_distance(
            init_state.position, self._env.current_episode.goals[0].position,
        )
        return init_distance

    def cand_dist_to_goal(self, angle: float, forward: float):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_distance = self._env.sim.geodesic_distance(
            post_state.position, self._env.current_episode.goals[0].position,
        )

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance

    def change_current_path(self, new_path: Any, collisions: Any):
        '''just for recording current path in high to low'''
        if 'current_path' not in self._env.current_episode.info.keys():
            self._env.current_episode.info['current_path'] = [np.array(self._env.current_episode.start_position)]
        self._env.current_episode.info['current_path'] += new_path
        if 'collisions' not in self._env.current_episode.info.keys():
            self._env.current_episode.info['collisions'] = []
        self._env.current_episode.info['collisions'] += collisions
        

@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations: Observations):
        return 0.0

    def get_done(self, observations: Observations):
        return self._env.episode_over

    def get_info(self, observations: Observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
