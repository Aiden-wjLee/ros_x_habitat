import numpy as np
from habitat.config import Config
from habitat.config.default import get_config
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from src.evaluators.evaluator import Evaluator


class MultiRobotHabitatSimEvaluator(Evaluator):
    """
    다중 로봇을 위한 Habitat Sim Evaluator
    """

    def __init__(
        self,
        config_paths: str,
        input_type: str,
        model_path: str,
        enable_physics: bool = False,
        num_robots: int = 2,
        robot_coordination_mode: str = "INDEPENDENT",
    ):
        super().__init__(config_paths, input_type, model_path, enable_physics)
        
        self.num_robots = num_robots
        self.robot_coordination_mode = robot_coordination_mode
        
        # 다중 로봇 설정 적용
        self._setup_multi_robot_config()

    def _setup_multi_robot_config(self):
        """
        다중 로봇을 위한 설정 적용
        """
        # AGENTS 리스트 업데이트
        agent_list = [f"AGENT_{i}" for i in range(self.num_robots)]
        self.config.PHYSICS_SIMULATOR.AGENTS = agent_list
        
        # 각 에이전트별 센서 설정
        for i in range(self.num_robots):
            agent_key = f"AGENT_{i}"
            
            # 에이전트별 센서 리스트
            sensors = [f"RGB_SENSOR_{i}", f"DEPTH_SENSOR_{i}"]
            setattr(self.config.PHYSICS_SIMULATOR, agent_key, Config())
            getattr(self.config.PHYSICS_SIMULATOR, agent_key).SENSORS = sensors
            getattr(self.config.PHYSICS_SIMULATOR, agent_key).ANGULAR_FRICTION = 0.0
            getattr(self.config.PHYSICS_SIMULATOR, agent_key).LINEAR_FRICTION = 0.0
            
            # 센서 설정 복사
            for sensor_type in ["RGB_SENSOR", "DEPTH_SENSOR"]:
                sensor_name = f"{sensor_type}_{i}"
                original_sensor = getattr(self.config.PHYSICS_SIMULATOR, sensor_type)
                setattr(self.config.PHYSICS_SIMULATOR, sensor_name, original_sensor.clone())

    @classmethod
    def compute_multi_robot_metrics(
        cls,
        dict_of_metrics_per_robot: Dict[int, Dict[str, Dict[str, float]]],
    ) -> Dict[str, float]:
        """
        다중 로봇의 메트릭을 종합하여 계산
        
        :param dict_of_metrics_per_robot: 로봇별 메트릭 딕셔너리
        :return: 종합된 메트릭
        """
        combined_metrics = {}
        
        # 각 로봇별 평균 메트릭 계산
        for robot_id, robot_metrics in dict_of_metrics_per_robot.items():
            avg_metrics = cls.compute_avg_metrics(robot_metrics)
            
            # 로봇별 메트릭 저장
            for metric_name, metric_value in avg_metrics.items():
                combined_metrics[f"robot_{robot_id}_{metric_name}"] = metric_value
        
        # 전체 로봇 평균 계산
        if dict_of_metrics_per_robot:
            all_metrics_by_type = defaultdict(list)
            
            for robot_metrics in dict_of_metrics_per_robot.values():
                robot_avg = cls.compute_avg_metrics(robot_metrics)
                for metric_name, metric_value in robot_avg.items():
                    all_metrics_by_type[metric_name].append(metric_value)
            
            # 전체 평균
            for metric_name, values in all_metrics_by_type.items():
                combined_metrics[f"overall_{metric_name}"] = np.mean(values)
                combined_metrics[f"std_{metric_name}"] = np.std(values)
        
        # 다중 로봇 특화 메트릭
        combined_metrics["num_robots"] = len(dict_of_metrics_per_robot)
        
        return combined_metrics

    @classmethod 
    def compute_coordination_metrics(
        cls,
        robot_trajectories: Dict[int, List[Tuple[float, float, float]]],
        min_distance_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        로봇 간 협조/충돌 메트릭 계산
        
        :param robot_trajectories: 로봇별 궤적 (시간별 위치)
        :param min_distance_threshold: 최소 거리 임계값
        :return: 협조 관련 메트릭
        """
        coordination_metrics = {}
        
        if len(robot_trajectories) < 2:
            return coordination_metrics
        
        robot_ids = list(robot_trajectories.keys())
        min_distances = []
        collision_count = 0
        
        # 모든 로봇 쌍에 대해 거리 계산
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                robot_i_traj = robot_trajectories[robot_ids[i]]
                robot_j_traj = robot_trajectories[robot_ids[j]]
                
                # 궤적 길이를 맞춤
                min_len = min(len(robot_i_traj), len(robot_j_traj))
                
                for t in range(min_len):
                    pos_i = np.array(robot_i_traj[t])
                    pos_j = np.array(robot_j_traj[t])
                    
                    distance = np.linalg.norm(pos_i - pos_j)
                    min_distances.append(distance)
                    
                    if distance < min_distance_threshold:
                        collision_count += 1
        
        if min_distances:
            coordination_metrics["min_inter_robot_distance"] = np.min(min_distances)
            coordination_metrics["avg_inter_robot_distance"] = np.mean(min_distances)
            coordination_metrics["collision_events"] = collision_count
            coordination_metrics["collision_rate"] = collision_count / len(min_distances)
        
        return coordination_metrics

    def evaluate_multi_robot_episode(
        self,
        episode_ids: List[str],
        scene_ids: List[str],
        agent_seeds: List[int],
        initial_positions: Optional[List[List[float]]] = None,
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        다중 로봇 에피소드 평가
        
        :param episode_ids: 로봇별 에피소드 ID 리스트
        :param scene_ids: 로봇별 씬 ID 리스트  
        :param agent_seeds: 로봇별 시드 리스트
        :param initial_positions: 로봇별 초기 위치
        :return: 종합된 평가 메트릭
        """
        
        # 로봇 수만큼 입력 확장
        if len(episode_ids) < self.num_robots:
            episode_ids = episode_ids * self.num_robots
        if len(scene_ids) < self.num_robots:
            scene_ids = scene_ids * self.num_robots
        if len(agent_seeds) < self.num_robots:
            agent_seeds = agent_seeds * self.num_robots
            
        robot_metrics = {}
        robot_trajectories = {}
        
        # 각 로봇별로 평가 실행
        for robot_id in range(self.num_robots):
            try:
                # 개별 로봇 평가 (기존 단일 로봇 평가 메서드 사용)
                single_robot_metrics = self._evaluate_single_robot(
                    robot_id=robot_id,
                    episode_id=episode_ids[robot_id],
                    scene_id=scene_ids[robot_id],
                    agent_seed=agent_seeds[robot_id],
                    initial_position=initial_positions[robot_id] if initial_positions else None
                )
                
                robot_metrics[robot_id] = single_robot_metrics["metrics"]
                robot_trajectories[robot_id] = single_robot_metrics.get("trajectory", [])
                
            except Exception as e:
                print(f"로봇 {robot_id} 평가 중 오류: {e}")
                robot_metrics[robot_id] = {}
                robot_trajectories[robot_id] = []
        
        # 다중 로봇 메트릭 계산
        combined_metrics = self.compute_multi_robot_metrics(robot_metrics)
        
        # 협조 메트릭 계산
        coordination_metrics = self.compute_coordination_metrics(robot_trajectories)
        combined_metrics.update(coordination_metrics)
        
        return {
            "combined_metrics": combined_metrics,
            "individual_robot_metrics": robot_metrics,
            "coordination_metrics": coordination_metrics
        }

    def _evaluate_single_robot(
        self,
        robot_id: int,
        episode_id: str,
        scene_id: str,
        agent_seed: int,
        initial_position: Optional[List[float]] = None,
    ) -> Dict:
        """
        단일 로봇 평가 (기존 평가 로직을 로봇 ID별로 적용)
        """
        # 이 부분은 기존 HabitatSimEvaluator의 평가 로직을 
        # 특정 로봇에 대해 실행하도록 구현해야 함
        
        # 예시 구현:
        metrics = {
            "distance_to_goal": np.random.random(),  # 실제 구현 필요
            "success": np.random.choice([0, 1]),     # 실제 구현 필요
            "spl": np.random.random(),               # 실제 구현 필요
        }
        
        trajectory = [
            (np.random.random(), np.random.random(), 0.0) 
            for _ in range(100)  # 실제 궤적 데이터로 교체 필요
        ]
        
        return {
            "metrics": {f"episode_{episode_id}": metrics},
            "trajectory": trajectory
        }

    def generate_multi_robot_videos(
        self,
        episode_ids: List[str],
        scene_ids: List[str],
        agent_seeds: List[int],
        video_dir: str = "videos/multi_robot/",
        *args,
        **kwargs,
    ) -> None:
        """
        다중 로봇 비디오 생성
        """
        for robot_id in range(self.num_robots):
            robot_video_dir = f"{video_dir}/robot_{robot_id}/"
            
            # 개별 로봇 비디오 생성 (기존 메서드 활용)
            self.generate_videos(
                episode_ids=[episode_ids[robot_id % len(episode_ids)]],
                scene_ids=[scene_ids[robot_id % len(scene_ids)]],
                agent_seed=agent_seeds[robot_id % len(agent_seeds)],
                video_dir=robot_video_dir,
                robot_id=robot_id,
                *args,
                **kwargs,
            )