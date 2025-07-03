#!/usr/bin/env python
# src/sims/multi_habitat_physics_simulator.py

from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Union,
    Dict,
)

import habitat_sim
import habitat_sim as hsim
import numpy as np
from gym import spaces
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.spaces import Space
from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSimSensor,
    overwrite_config,
)
from numpy import ndarray

from src.sims.physics_simulator import PhysicsSimulator


@registry.register_simulator(name="Sim-Multi-Phys")
class MultiHabitatPhysicsSim(PhysicsSimulator, Simulator):
    r"""Multi-agent physics simulator wrapper over habitat-sim.
    Supports multiple agents with independent control and observations.

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        self.habitat_config = config
        self.num_agents = len(config.AGENTS)
        
        # Initialize sensor suites for each agent
        self._sensor_suites = {}
        self._action_spaces = {}
        
        for agent_idx, agent_name in enumerate(config.AGENTS):
            agent_config = getattr(config, agent_name)
            
            sim_sensors = []
            for sensor_name in agent_config.SENSORS:
                sensor_cfg = getattr(self.habitat_config, sensor_name)
                sensor_type = registry.get_sensor(sensor_cfg.TYPE)

                assert sensor_type is not None, f"invalid sensor type {sensor_cfg.TYPE}"
                sim_sensors.append(sensor_type(sensor_cfg))

            self._sensor_suites[agent_idx] = SensorSuite(sim_sensors)
            self._action_spaces[agent_idx] = spaces.Discrete(
                len(self.create_sim_config(self._sensor_suites[agent_idx]).agents[0].action_space)
            )

        # Create sim config using the first agent's sensors (can be modified for different configs per agent)
        self.sim_config = self.create_sim_config(self._sensor_suites[0])
        self._current_scene = self.sim_config.sim_cfg.scene_id
        
        # Initialize the base simulator
        super().__init__(self.sim_config)
        
        self._prev_sim_obs: Dict[int, Optional[Observations]] = {
            i: None for i in range(self.num_agents)
        }

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        """Create simulation configuration with multi-agent support"""
        sim_config = habitat_sim.SimulatorConfiguration()
        
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
            
        overwrite_config(
            config_from=self.habitat_config.HABITAT_SIM_V0,
            config_to=sim_config,
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_id = self.habitat_config.SCENE

        # Create agent configurations for all agents
        agent_configs = []
        
        for agent_idx, agent_name in enumerate(self.habitat_config.AGENTS):
            agent_config = habitat_sim.AgentConfiguration()
            agent_habitat_config = getattr(self.habitat_config, agent_name)
            
            overwrite_config(
                config_from=agent_habitat_config,
                config_to=agent_config,
                ignore_keys={
                    "is_set_start_state",
                    "sensors",
                    "start_position",
                    "start_rotation",
                },
            )

            # Sensor specifications for this agent
            sensor_specifications = []
            agent_sensor_suite = self._sensor_suites[agent_idx]
            
            for sensor in agent_sensor_suite.sensors.values():
                assert isinstance(sensor, HabitatSimSensor)
                sim_sensor_cfg = sensor._get_default_spec()
                
                overwrite_config(
                    config_from=sensor.config,
                    config_to=sim_sensor_cfg,
                    ignore_keys=sensor._config_ignore_keys,
                    trans_dict={
                        "sensor_model_type": lambda v: getattr(
                            habitat_sim.FisheyeSensorModelType, v
                        ),
                        "sensor_subtype": lambda v: getattr(habitat_sim.SensorSubType, v),
                    },
                )
                
                # Add agent index to sensor UUID to make it unique
                sim_sensor_cfg.uuid = f"agent_{agent_idx}_{sensor.uuid}"
                sim_sensor_cfg.resolution = list(sensor.observation_space.shape[:2])
                sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
                sim_sensor_cfg.gpu2gpu_transfer = self.habitat_config.HABITAT_SIM_V0.GPU_GPU
                sensor_specifications.append(sim_sensor_cfg)

            agent_config.sensor_specifications = sensor_specifications
            agent_config.action_space = registry.get_action_space_configuration(
                self.habitat_config.ACTION_SPACE_CONFIG
            )(self.habitat_config).get()

            # Set different starting positions for agents to avoid overlap
            if hasattr(agent_habitat_config, 'START_POSITION'):
                agent_config.start_position = agent_habitat_config.START_POSITION
            else:
                # Default positions spread out in a line
                agent_config.start_position = [agent_idx * 2.0, 0.0, 0.0]
                
            if hasattr(agent_habitat_config, 'START_ROTATION'):
                agent_config.start_rotation = agent_habitat_config.START_ROTATION
            else:
                agent_config.start_rotation = [0.0, 0.0, 0.0, 1.0]

            agent_configs.append(agent_config)

        return habitat_sim.Configuration(sim_config, agent_configs)

    @property
    def sensor_suite(self) -> Dict[int, SensorSuite]:
        """Return sensor suites for all agents"""
        return self._sensor_suites

    @property
    def action_space(self) -> Dict[int, Space]:
        """Return action spaces for all agents"""
        return self._action_spaces

    def get_agent_sensor_suite(self, agent_id: int) -> SensorSuite:
        """Get sensor suite for specific agent"""
        return self._sensor_suites.get(agent_id, self._sensor_suites[0])

    def get_agent_action_space(self, agent_id: int) -> Space:
        """Get action space for specific agent"""
        return self._action_spaces.get(agent_id, self._action_spaces[0])

    def _update_agents_state(self) -> bool:
        """Update states for all agents"""
        is_updated = False
        
        for agent_id, agent_name in enumerate(self.habitat_config.AGENTS):
            agent_cfg = getattr(self.habitat_config, agent_name)
            
            if hasattr(agent_cfg, 'IS_SET_START_STATE') and agent_cfg.IS_SET_START_STATE:
                start_position = getattr(agent_cfg, 'START_POSITION', [agent_id * 2.0, 0.0, 0.0])
                start_rotation = getattr(agent_cfg, 'START_ROTATION', [0.0, 0.0, 0.0, 1.0])
                
                self.set_agent_state(
                    start_position,
                    start_rotation,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def reset(self) -> Dict[int, Observations]:
        """Reset environment and return observations for all agents"""
        sim_obs = super().reset()
        
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        # Process observations for each agent
        agent_observations = {}
        
        for agent_id in range(self.num_agents):
            if isinstance(sim_obs, dict):
                agent_sim_obs = sim_obs.get(agent_id, sim_obs)
            else:
                agent_sim_obs = sim_obs
                
            self._prev_sim_obs[agent_id] = agent_sim_obs
            sensor_suite = self._sensor_suites[agent_id]
            agent_observations[agent_id] = sensor_suite.get_observations(agent_sim_obs)

        return agent_observations

    def step(self, action: Union[str, int], agent_id: int = 0) -> Dict[int, Observations]:
        """Step environment for specific agent"""
        # Set the agent we want to step
        original_default_agent = self._default_agent_id
        self._default_agent_id = agent_id
        
        sim_obs = super().step(action)
        
        # Restore original default agent
        self._default_agent_id = original_default_agent
        
        # Process observations for all agents
        agent_observations = {}
        
        for aid in range(self.num_agents):
            if isinstance(sim_obs, dict):
                agent_sim_obs = sim_obs.get(aid, sim_obs)
            else:
                agent_sim_obs = sim_obs
                
            self._prev_sim_obs[aid] = agent_sim_obs
            sensor_suite = self._sensor_suites[aid]
            agent_observations[aid] = sensor_suite.get_observations(agent_sim_obs)

        return agent_observations

    def step_physics(
        self, 
        agent_object: hsim.physics.ManagedRigidObject, 
        time_step: float,
        agent_id: int = 0
    ) -> Dict[int, Observations]:
        """Step physics for specific agent"""
        original_default_agent = self._default_agent_id
        self._default_agent_id = agent_id
        
        sim_obs = super().step_physics(agent_object, time_step)
        
        self._default_agent_id = original_default_agent
        
        # Process observations for all agents
        agent_observations = {}
        
        for aid in range(self.num_agents):
            if isinstance(sim_obs, dict):
                agent_sim_obs = sim_obs.get(aid, sim_obs)
            else:
                agent_sim_obs = sim_obs
                
            self._prev_sim_obs[aid] = agent_sim_obs
            sensor_suite = self._sensor_suites[aid]
            agent_observations[aid] = sensor_suite.get_observations(agent_sim_obs)

        return agent_observations

    def set_agent_velocities(self, linear_vel: np.ndarray, angular_vel: np.ndarray, agent_id: int = 0):
        """Set velocities for specific agent"""
        if hasattr(self, '_env') and hasattr(self._env, 'set_agent_velocities'):
            self._env.set_agent_velocities(linear_vel, angular_vel, agent_id)
        else:
            # Fallback to basic agent state setting
            agent = self.get_agent(agent_id)
            current_state = agent.get_state()
            
            # Simple velocity-based position update (this is a simplified approach)
            dt = 0.1  # Assuming 10Hz update rate
            
            # Update position based on linear velocity
            new_position = current_state.position + linear_vel * dt
            
            # Update rotation based on angular velocity
            current_rotation = current_state.rotation
            angular_magnitude = np.linalg.norm(angular_vel)
            
            if angular_magnitude > 0:
                axis = angular_vel / angular_magnitude
                angle = angular_magnitude * dt
                
                # Convert axis-angle to quaternion
                from scipy.spatial.transform import Rotation as R
                rotation_delta = R.from_rotvec(axis * angle)
                current_rot = R.from_quat([current_rotation.x, current_rotation.y, 
                                         current_rotation.z, current_rotation.w])
                new_rot = rotation_delta * current_rot
                new_quat = new_rot.as_quat()
                new_rotation = [new_quat[0], new_quat[1], new_quat[2], new_quat[3]]
            else:
                new_rotation = [current_rotation.x, current_rotation.y, 
                               current_rotation.z, current_rotation.w]
            
            self.set_agent_state(new_position.tolist(), new_rotation, agent_id)

    def render(self, mode: str = "rgb", agent_id: int = 0) -> Any:
        """Render from specific agent's perspective"""
        sim_obs = self.get_sensor_observations(agent_ids=[agent_id])
        
        if isinstance(sim_obs, dict):
            agent_sim_obs = sim_obs.get(agent_id, sim_obs)
        else:
            agent_sim_obs = sim_obs
            
        sensor_suite = self._sensor_suites[agent_id]
        observations = sensor_suite.get_observations(agent_sim_obs)

        output = observations.get(mode)
        assert output is not None, f"mode {mode} sensor is not active for agent {agent_id}"
        
        if not isinstance(output, np.ndarray):
            output = output.to("cpu").numpy()

        return output

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        """Get state for specific agent"""
        return self.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        """Set state for specific agent"""
        agent = self.get_agent(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
        agent_id: int = 0,
    ) -> Optional[Observations]:
        """Get observations at specific position for specific agent"""
        current_state = self.get_agent_state(agent_id)
        
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(position, rotation, agent_id, reset_sensors=False)

        if success:
            sim_obs = self.get_sensor_observations(agent_ids=[agent_id])
            
            if isinstance(sim_obs, dict):
                agent_sim_obs = sim_obs.get(agent_id, sim_obs)
            else:
                agent_sim_obs = sim_obs
                
            self._prev_sim_obs[agent_id] = agent_sim_obs
            sensor_suite = self._sensor_suites[agent_id]
            observations = sensor_suite.get_observations(agent_sim_obs)
            
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    agent_id,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    @property
    def previous_step_collided(self):
        """Check if any agent collided in the previous step"""
        collisions = {}
        for agent_id in range(self.num_agents):
            if self._prev_sim_obs[agent_id] is not None:
                collisions[agent_id] = self._prev_sim_obs[agent_id].get("collided", False)
            else:
                collisions[agent_id] = False
        return collisions

    def agent_previous_step_collided(self, agent_id: int) -> bool:
        """Check if specific agent collided in the previous step"""
        if self._prev_sim_obs.get(agent_id) is not None:
            return self._prev_sim_obs[agent_id].get("collided", False)
        return False

    def get_all_agent_states(self) -> Dict[int, habitat_sim.AgentState]:
        """Get states for all agents"""
        states = {}
        for agent_id in range(self.num_agents):
            states[agent_id] = self.get_agent_state(agent_id)
        return states

    def set_all_agent_states(self, states: Dict[int, tuple]) -> Dict[int, bool]:
        """Set states for all agents
        
        Args:
            states: Dict mapping agent_id to (position, rotation) tuple
            
        Returns:
            Dict mapping agent_id to success boolean
        """
        results = {}
        for agent_id, (position, rotation) in states.items():
            results[agent_id] = self.set_agent_state(position, rotation, agent_id)
        return results

    def reconfigure(self, habitat_config: Config) -> None:
        """Reconfigure the multi-agent simulator"""
        self.habitat_config = habitat_config
        self.num_agents = len(habitat_config.AGENTS)
        
        # Reinitialize sensor suites and action spaces
        self._sensor_suites = {}
        self._action_spaces = {}
        
        for agent_idx, agent_name in enumerate(habitat_config.AGENTS):
            agent_config = getattr(habitat_config, agent_name)
            
            sim_sensors = []
            for sensor_name in agent_config.SENSORS:
                sensor_cfg = getattr(self.habitat_config, sensor_name)
                sensor_type = registry.get_sensor(sensor_cfg.TYPE)
                assert sensor_type is not None
                sim_sensors.append(sensor_type(sensor_cfg))

            self._sensor_suites[agent_idx] = SensorSuite(sim_sensors)
            self._action_spaces[agent_idx] = spaces.Discrete(
                len(self.create_sim_config(self._sensor_suites[agent_idx]).agents[0].action_space)
            )

        self.sim_config = self.create_sim_config(self._sensor_suites[0])
        self._current_scene = habitat_config.SCENE
        
        # Reinitialize previous observations
        self._prev_sim_obs = {i: None for i in range(self.num_agents)}
        
        self.close()
        super().reconfigure(self.sim_config)
        self._update_agents_state()