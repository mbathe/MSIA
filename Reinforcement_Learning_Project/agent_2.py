"""
Environnement Gymnasium pour un robot Kuka IIWA devant atteindre un cube cible
parmi plusieurs, en utilisant PyBullet. Inclut des fonctions pour entraîner
un agent RL (Stable Baselines3) et charger un agent entraîné.
"""

import os
import time
import math
import random
from typing import Optional, Dict, Tuple, Any, List

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- Constantes ---
DEFAULT_SUCCESS_EPS = 0.07  # Rayon de succès pour le cube CIBLE [m]
DEFAULT_SUBSTEPS = 30       # Pas PyBullet par step RL
DEFAULT_EE_LINK_INDEX = 6 # Index du link de l'effecteur final (lbr_iiwa_link_7)
DEFAULT_NUM_CUBES = 3       # Nombre de cubes par défaut
DEFAULT_MIN_DIST_BETWEEN_CUBES = 0.2 # Dist. min. entre centres des cubes spawnés
DEFAULT_NON_TARGET_ZONE_EPS = 0.07 # Rayon zone exclusion cubes NON-CIBLES [m]

# --- Environnement KukaMultiCubeReachEnv ---
class KukaMultiCubeReachEnv(gym.Env):
    """
    Environnement Gymnasium simulant un Kuka IIWA pour une tâche de Reach multi-cubes.

    L'objectif est d'amener l'effecteur final du robot près d'un cube cible
    désigné aléatoirement parmi plusieurs cubes présents dans l'environnement,
    tout en évitant de déplacer les autres cubes ou de s'en approcher trop près.

    **Espace d'observation:** (détaillé dans __init__)
        - Positions des joints
        - Vitesses des joints
        - Position absolue du cube cible
        - Position relative EE -> cube cible
        - Vitesse linéaire de l'EE
        - Index normalisé du cube cible

    **Espace d'action:** (détaillé dans __init__)
        - Commande en vitesse pour chaque joint, normalisée dans [-1, 1].

    **Récompense:**
        - Négative proportionnelle à la distance EE-Cible.
        - Bonus élevé en cas de succès (distance < success_eps).
        - Bonus de shaping basé sur l'amélioration de la distance minimale atteinte.
        - Pénalité si un cube non-cible est déplacé.
        - Pénalité si l'EE entre dans la zone d'exclusion d'un cube non-cible.
        - Pénalité en cas de collision robot-sol.
        - Pénalité proportionnelle à la magnitude de l'action.
        - Petite pénalité de temps à chaque pas.

    **Terminaison (terminated):**
        - Succès (EE proche du cube cible).
        - Collision robot-sol.
        # (Optionnel, désactivé par défaut) Déplacement excessif d'un cube non-cible.
        # (Optionnel, désactivé par défaut) Violation zone d'exclusion non-cible.

    **Troncation (truncated):**
        - Nombre maximum de pas atteint (max_episode_steps).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 num_cubes: int = DEFAULT_NUM_CUBES,
                 cube_size: float = 0.04,
                 r_min: float = 0.30,
                 r_max: float = 0.82,
                 max_episode_steps: int = 300,
                 substeps: int = DEFAULT_SUBSTEPS,
                 success_eps: float = DEFAULT_SUCCESS_EPS,
                 ee_link_index: int = DEFAULT_EE_LINK_INDEX,
                 joint_force: float = 100.0,
                 cube_move_penalty_coeff: float = 5.0,
                 cube_move_threshold: float = 0.01,
                 non_target_zone_penalty_coeff: float = 2.0,
                 non_target_zone_eps: float = DEFAULT_NON_TARGET_ZONE_EPS,
                 action_penalty_coeff: float = 0.001,
                 max_ee_velocity: float = 2.0,
                 min_dist_between_cubes: float = DEFAULT_MIN_DIST_BETWEEN_CUBES,
                 continue_training_episodes: bool = False
                 ):
        super().__init__()
        self.render_mode = render_mode
        # 'renders' est déprécié au profit de render_mode, on garde juste render_mode
        self.is_rendering = self.render_mode in ["human", "rgb_array"]

        self.continue_training_episodes = continue_training_episodes

        # --- Paramètres spécifiques multi-cubes ---
        if num_cubes <= 0:
            raise ValueError("num_cubes doit être positif.")
        self.num_cubes = num_cubes
        self.min_dist_between_cubes = min_dist_between_cubes
        self.non_target_zone_eps = non_target_zone_eps
        self.non_target_zone_penalty_coeff = non_target_zone_penalty_coeff
        self.cube_move_penalty_coeff = cube_move_penalty_coeff
        self.cube_move_threshold = cube_move_threshold

        # --- Autres paramètres ---
        self.cube_size = cube_size
        self.r_min = r_min
        self.r_max = r_max
        self._max_episode_steps = max_episode_steps # Renommé pour éviter conflit
        self.substeps = substeps
        self.success_eps = success_eps
        self.ee_link_index = ee_link_index
        self.joint_force = joint_force
        self.max_ee_velocity = max_ee_velocity
        self.action_penalty_coeff = action_penalty_coeff

        # Initialisation PyBullet (connexion différée à reset pour environnements vectorisés)
        self.physicsClient = None
        self.plane_id = None
        self.kuka_id = None
        self.timeStep = 1. / 240.

        # Option du Reset 
        self._next_reset_options: Optional[Dict] = None

        # Propriétés du robot (initialisées après chargement)
        self.num_arm_joints = 7 # Pour Kuka IIWA typique
        self.max_joint_velocity = np.zeros(self.num_arm_joints, dtype=np.float32)
        self.joint_ranges = []
        self.joint_lower_limits = np.zeros(self.num_arm_joints, dtype=np.float32)
        self.joint_upper_limits = np.zeros(self.num_arm_joints, dtype=np.float32)

        # --- Espace d'action (contrôle en VITESSE) ---
        # L'action est toujours entre [-1, 1] par joint
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_arm_joints,), dtype=np.float32)

        # --- Espace d'observation ---
        # Sera défini précisément dans _setup_simulation basé sur les limites réelles
        obs_dim = self.num_arm_joints * 2 + 3 + 3 + 3 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Variables d'état
        self.current_step = 0
        self.cube_ids: List[int] = []
        self.cube_ref_positions: List[np.ndarray] = []
        self.target_cube_index: int = -1
        self.best_distance = np.inf


    def _setup_simulation(self):
        """Initialise la simulation PyBullet si elle ne l'est pas déjà."""
        if self.physicsClient is not None:
            return # Déjà initialisé

        connection_mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self.physicsClient = p.connect(connection_mode)
        if self.physicsClient < 0:
             raise RuntimeError("Impossible de se connecter à PyBullet.")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timeStep)

        # Chargement des éléments
        self.plane_id = p.loadURDF("plane.urdf")
        # Note: Assurez-vous que le chemin vers kuka_iiwa est correct
        # ou que le modèle est dans un chemin recherché par pybullet_data
        try:
            self.kuka_id = p.loadURDF("kuka_iiwa/model_vr_limits.urdf", [0., 0., 0.], [0, 0, 0, 1], useFixedBase=True)
        except p.error as e:
            print("\n" + "="*30 + " ERREUR " + "="*30)
            print("Impossible de charger 'kuka_iiwa/model_vr_limits.urdf'.")
            print("Assurez-vous que PyBullet peut trouver ce fichier.")
            print("Il est peut-être nécessaire d'installer des assets supplémentaires")
            print("ou de les placer dans un répertoire accessible.")
            print(f"Erreur PyBullet: {e}")
            print("="*68 + "\n")
            raise

        actual_num_joints = p.getNumJoints(self.kuka_id)
        if actual_num_joints != self.num_arm_joints:
             print(f"Avertissement: Le modèle Kuka chargé a {actual_num_joints} joints, "
                   f"mais {self.num_arm_joints} étaient attendus. Ajustement...")
             self.num_arm_joints = actual_num_joints
             # Redéfinir action_space et recalculer obs_dim si nécessaire
             self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_arm_joints,), dtype=np.float32)


        if self.ee_link_index >= self.num_arm_joints:
             print(f"Avertissement: ee_link_index ({self.ee_link_index}) >= num_arm_joints ({self.num_arm_joints}). Utilisation du dernier indice de lien.")
             self.ee_link_index = self.num_arm_joints - 1

        # Récupérer les limites réelles des joints et définir l'espace d'observation
        self._update_joint_limits()
        self._define_observation_space()


    def _update_joint_limits(self):
        """ Récupère les limites des joints depuis le modèle chargé. """
        self.max_joint_velocity = []
        self.joint_ranges = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        for i in range(self.num_arm_joints):
            joint_info = p.getJointInfo(self.kuka_id, i)
            # Indices: 8=lowerLimit, 9=upperLimit, 11=maxVelocity
            lower, upper = joint_info[8], joint_info[9]
             # Ignorer les joints fixes ou sans limites définies correctement
            if lower > upper:
                print(f"Joint {i} ({joint_info[1]}) a des limites invalides ({lower}, {upper}). Utilisation de limites par défaut [0,0].")
                lower, upper = 0.0, 0.0
            self.joint_ranges.append((lower, upper))
            self.joint_lower_limits.append(lower)
            self.joint_upper_limits.append(upper)
             # S'assurer que maxVelocity est positif, sinon utiliser une valeur par défaut raisonnable
            max_vel = joint_info[11]
            if max_vel <= 0:
                # print(f"Joint {i} ({joint_info[1]}) a une maxVelocity <= 0 ({max_vel}). Utilisation de 2.0 rad/s.")
                max_vel = 2.0 # Valeur par défaut
            self.max_joint_velocity.append(max_vel)

        self.joint_lower_limits = np.array(self.joint_lower_limits, dtype=np.float32)
        self.joint_upper_limits = np.array(self.joint_upper_limits, dtype=np.float32)
        self.max_joint_velocity = np.array(self.max_joint_velocity, dtype=np.float32)


    def _define_observation_space(self):
        """ Définit l'espace d'observation basé sur les limites des joints. """
        # pos joints + vel joints + pos CUBE CIBLE + pos relative EE->CIBLE + vel EE + index CIBLE (normalisé)
        obs_dim = self.num_arm_joints * 2 + 3 + 3 + 3 + 1
        distance_max =  self.r_max + 1
        low_obs = np.concatenate([
            self.joint_lower_limits,                        # joint pos
            -self.max_joint_velocity * 2,                   # joint velocities (estimation large)
            -distance_max * np.ones(3) * 1.5,                 # target cube position x,y,z
            -2 * distance_max * np.ones(3) * 1.5,             # relative EE→target x,y,z
            -self.max_ee_velocity * np.ones(3) * 1.5,       # EE velocity vx,vy,vz
            [0.0]                                           # target cube index (normalisé)
        ])
        high_obs = np.concatenate([
            self.joint_upper_limits,
            self.max_joint_velocity * 2,
            distance_max * np.ones(3) * 1.5,
            2 * distance_max * np.ones(3) * 1.5,
            self.max_ee_velocity * np.ones(3) * 1.5,
            [1.0] # Normalisé entre 0 et 1 (si num_cubes > 1)
        ])
        self.observation_space = spaces.Box(low=low_obs,
                                            high=high_obs,
                                            shape=(obs_dim,),
                                            dtype=np.float32)


    def set_next_reset_options(self, options: Dict):
        """
        Stocke les options qui seront utilisées lors du prochain appel à reset().
        Ceci est utilisé pour passer des options à travers les wrappers VecEnv.
        """
        # print(f"DEBUG: Setting next reset options: {options}") # Debug
        self._next_reset_options = options


    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Réinitialise l'environnement à un état initial.

        Args:
            seed: Graine pour la génération de nombres aléatoires.
            options: Dictionnaire optionnel pouvant contenir :
                - 'mode' (str): 'train' (défaut) ou 'eval'.
                - 'target_index' (int): Si mode='eval', spécifie l'index du cube cible.
                - 'start_from_previous_state' (bool): Si True (et pas le premier reset),
                ne réinitialise pas la position du robot ni les cubes,
                change seulement la cible et met à jour les positions de référence.
                Défaut à False (reset complet).


        Returns:
            Tuple contenant l'observation initiale et le dictionnaire d'info.
        """
        super().reset(seed=seed)


        # Utiliser les options stockées si disponibles, sinon celles passées en argument
        options = self._next_reset_options if self._next_reset_options is not None else options
        if options is None:
            options = {} # Assurer que c'est un dictionnaire

        # Réinitialiser les options stockées après les avoir lues
        self._next_reset_options = None

        # Déterminer le mode (défaut 'train')
        if options is None: options = {}
        mode = options.get('mode', 'train')

        start_from_previous_state = self.continue_training_episodes  #options.get('start_from_previous_state', False)


        # Initialiser la simulation PyBullet si ce n'est pas déjà fait
        self._setup_simulation()

        # On vérifie si self.cube_ids a été initialisé et n'est pas vide.
        is_first_reset = not hasattr(self, 'cube_ids') or not self.cube_ids
        # print('start_from_previous_state',start_from_previous_state)


        if not start_from_previous_state or is_first_reset:
            # --- Reset du robot ---
            # Utiliser les limites récupérées pour la position initiale
            # initial_joint_positions = (self.joint_lower_limits + self.joint_upper_limits) / 2.0
            for i in range(self.num_arm_joints):
                initial_joint_positions =  random.uniform(self.joint_lower_limits[i], self.joint_upper_limits[i])
                # Utiliser la map des indices mobiles
                p.resetJointState(self.kuka_id, i, targetValue=initial_joint_positions, targetVelocity=0.0)
                # Désactiver le moteur au cas où il serait actif (pour éviter forces résiduelles)
                p.setJointMotorControl2(self.kuka_id, i, p.VELOCITY_CONTROL, force=0)


            # --- Suppression des anciens cubes ---
            for cube_id in self.cube_ids:
                try:
                    p.removeBody(cube_id)
                except p.error:
                    pass
            self.cube_ids = []
            self.cube_ref_positions = []
            self.target_cube_index = -1
            self._current_num_cubes_spawned = 0

            # --- Spawn de N nouveaux cubes ---
            spawn_successful = False
            max_reset_trials = 20
            for _ in range(max_reset_trials):
                if self._spawn_n_cubes(self.num_cubes, self.r_min, self.r_max, self.cube_size, self.min_dist_between_cubes):
                    spawn_successful = True
                    # Stocker le nombre réel de cubes spawnés
                    self._current_num_cubes_spawned = len(self.cube_ids)
                    break
                else:
                    # Nettoyer avant nouvelle tentative
                    for cube_id in self.cube_ids:
                        try: p.removeBody(cube_id)
                        except p.error: pass
                    self.cube_ids = []
                    self.cube_ref_positions = []

            if not spawn_successful or self._current_num_cubes_spawned == 0:
                raise RuntimeError(f"Impossible de spawner au moins 1 cube atteignable après {max_reset_trials} tentatives.")

                    # --- Enregistrer les positions initiales de TOUS les cubes spawnés ---

            self.cube_ref_positions = []
            for cube_id in self.cube_ids:
                try:
                    pos, _ = p.getBasePositionAndOrientation(cube_id)
                    self.cube_ref_positions.append(np.array(pos))
                except p.error as e:
                    print(f"Erreur PyBullet en récupérant la position initiale du cube {cube_id}: {e}")
                    self.cube_ref_positions.append(np.zeros(3))


        
        self.current_step = 0
        self.best_distance = np.inf
        # --- Choix du cube cible selon le mode ---
        if mode == 'eval':
            if 'target_index' not in options:
                raise ValueError("Le mode 'eval' nécessite 'target_index' dans les options de reset.")
            target_index_req = options['target_index']
            if not isinstance(target_index_req, int) or not (0 <= target_index_req < self._current_num_cubes_spawned):
                raise ValueError(f"Index cible invalide ({target_index_req}) fourni en mode 'eval'. "
                                 f"Doit être entre 0 et {self._current_num_cubes_spawned - 1}.")
            self.target_cube_index = target_index_req
            print(f"[Eval Mode] Target cube index set to: {self.target_cube_index}")

        elif mode == 'train':
            # Choix aléatoire parmi les cubes spawnés
            self.target_cube_index = self.np_random.integers(0, self._current_num_cubes_spawned)
            # print(f"[Train Mode] Random target cube index: {self.target_cube_index}")
        else:
            raise ValueError(f"Mode inconnu '{mode}' reçu dans les options de reset. Utiliser 'train' ou 'eval'.")


        # --- NOUVEAU / MODIFICATION: Mettre à jour les positions de référence SEULEMENT si on continue ---
        if start_from_previous_state and not is_first_reset:
            # print("DEBUG: Updating reference positions to current cube positions.") # Pour Debug
            new_ref_positions = []
            for cube_id in self.cube_ids:
                try:
                    pos, _ = p.getBasePositionAndOrientation(cube_id)
                    new_ref_positions.append(np.array(pos))
                except p.error as e:
                    print(f"Erreur PyBullet getBasePositionAndOrientation pour ref cube {cube_id}: {e}")
                    # Tentative de fallback vers l'ancienne position si possible
                    try:
                        old_index = self.cube_ids.index(cube_id) # Récupérer l'index actuel
                        if old_index < len(self.cube_ref_positions): # Vérifier si une ancienne ref existe
                            print(f"    -> Conservation de l'ancienne position de référence pour le cube {cube_id}")
                            new_ref_positions.append(self.cube_ref_positions[old_index])
                        else:
                            print(f"    -> Position de référence non trouvée pour le cube {cube_id}, utilisation de [0,0,0].")
                            new_ref_positions.append(np.zeros(3)) # Fallback
                    except (ValueError, IndexError): # Si index non trouvé ou hors limites
                        print(f"    -> Fallback: Utilisation de [0,0,0] comme référence pour le cube {cube_id}.")
                        new_ref_positions.append(np.zeros(3)) # Fallback ultime

            self.cube_ref_positions = new_ref_positions # Mettre à jour avec les positions actuelles


        # Stabilisation (quelques pas de simulation sans action)
        for _ in range(50): 
             p.stepSimulation()

        observation = self._get_obs()
        info = self._get_info()
        if "distance_to_target" in info:
            self.best_distance = info["distance_to_target"]
        else:
            self.best_distance = np.inf

        # Ajouter le mode et l'index cible à l'info retournée
        info['reset_mode'] = mode
        info['target_cube_index'] = self.target_cube_index
        info['partial_reset'] = start_from_previous_state and not is_first_reset

        # print(f"--- RESET FINISHED  ---")
        return observation, info
    

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Met à l'échelle l'action de [-1, 1] vers les vitesses cibles des joints."""
        return action * self.max_joint_velocity


    def _is_pose_reachable(self, target_pos, target_orn=None) -> bool:
        """ Vérifie si une pose est atteignable par cinématique inverse. """
        # Vérifier que les limites sont définies
        if not hasattr(self, 'joint_lower_limits') or len(self.joint_lower_limits) == 0:
             print("Avertissement: Limites de joints non définies pour IK.")
             return False

        # Définir une orientation par défaut (EE vers le bas) si non fournie
        if target_orn is None:
             target_orn = p.getQuaternionFromEuler([0, -math.pi, 0]) # Pointant vers le bas

        # S'assurer que les limites sont passées en listes
        lower_limits = self.joint_lower_limits.tolist()
        upper_limits = self.joint_upper_limits.tolist()
        joint_range_val = (self.joint_upper_limits - self.joint_lower_limits).tolist()
        rest_poses = ((self.joint_lower_limits + self.joint_upper_limits) / 2.0).tolist()

        try:
             ik_solution = p.calculateInverseKinematics(
                 self.kuka_id, self.ee_link_index, target_pos, targetOrientation=target_orn,
                 lowerLimits=lower_limits,
                 upperLimits=upper_limits,
                 jointRanges=joint_range_val,
                 restPoses=rest_poses,
                 maxNumIterations=50, # Augmenter un peu si nécessaire
                 residualThreshold=1e-4 # Seuil acceptable pour la solution
             )
             # Vérifier si une solution a été retournée et si elle est valide (pas None et longueur correcte)
             return ik_solution is not None and len(ik_solution) == self.num_arm_joints

        except Exception as e:
             # print(f"Erreur IK pour {target_pos}: {e}") # Debug
             return False


    def _spawn_n_cubes(
            self,
            n: int,
            r_min: float,
            r_max: float,
            size: float,
            min_dist_between: float,
            height_offset: float = 0.0,
            max_trials_per_cube: int = 100
        ) -> bool:
        """
        Tente de faire apparaître N cubes atteignables et suffisamment espacés.
        Retourne True si succès, False sinon. Remplit self.cube_ids.
        """
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # S'assurer que le chemin est défini
        default_edge = 0.05 # Taille du cube dans cube_small.urdf
        scale = size / default_edge if default_edge > 0 else 1.0
        spawned_positions = []
        current_cube_ids = [] # Utiliser une liste temporaire

        for i in range(n): # Pour chaque cube à spawner
            cube_placed = False
            for trial in range(max_trials_per_cube):
                # Échantillonnage dans la couronne (utiliser self.np_random)
                radius = self.np_random.uniform(r_min, r_max)
                theta = self.np_random.uniform(np.pi/2, 3 * np.pi/2) # entre  - Y et Y (coté gauche)
                x, y = radius * np.cos(theta), radius * np.sin(theta)
                z = height_offset + size / 2.0
                pos = np.array([x, y, z])

                # 1. Vérifier l'atteignabilité (avec orientation par défaut)
                if not self._is_pose_reachable(pos):
                    continue # Essayer une autre position

                # 2. Vérifier la distance aux cubes déjà placés
                too_close = False
                for existing_pos in spawned_positions:
                    if np.linalg.norm(pos - existing_pos) < min_dist_between:
                        too_close = True
                        break
                if too_close:
                    continue # Essayer une autre position

                # Si la position est valide (atteignable et pas trop proche)
                try:
                    # Utiliser une orientation nulle (par défaut)
                    orientation = p.getQuaternionFromEuler([0,0,0])
                    cube_id = p.loadURDF(
                        "cube_small.urdf",
                        basePosition=pos.tolist(),
                        baseOrientation=orientation,
                        globalScaling=scale,
                        useFixedBase=False
                    )
                    current_cube_ids.append(cube_id)
                    spawned_positions.append(pos)
                    cube_placed = True
                    break # Passer au cube suivant
                except p.error as e:
                     # print(f"Erreur Pybullet lors du chargement du cube ({i+1}/{n}, essai {trial+1}): {e}")
                     # Essayer une autre position si le chargement échoue
                     continue

            if not cube_placed:
                # print(f"Impossible de placer le cube {i + 1} après {max_trials_per_cube} essais.")
                # Nettoyer les cubes déjà créés pour cette tentative de reset
                for c_id in current_cube_ids:
                    try: p.removeBody(c_id)
                    except p.error: pass
                return False # Échec du spawn de tous les cubes

        # Si tout s'est bien passé, assigner la liste temporaire à self.cube_ids
        self.cube_ids = current_cube_ids
        return True


    def _get_obs(self) -> np.ndarray:
        """Récupère l'observation actuelle basée sur le cube CIBLE."""
        # Vérifier si l'environnement est prêt
        if self.kuka_id is None or self.target_cube_index < 0 or self.target_cube_index >= len(self.cube_ids):
             print("Avertissement: _get_obs appelé avant initialisation complète ou avec index cible invalide.")
             # Retourner une observation nulle de la bonne taille
             return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        try:
            # États des joints
            joint_states = p.getJointStates(self.kuka_id, range(self.num_arm_joints))
            joint_pos = np.array([state[0] for state in joint_states], dtype=np.float32)
            joint_vel = np.array([state[1] for state in joint_states], dtype=np.float32)

            # État de l'effecteur final (EE)
            ee_state = p.getLinkState(self.kuka_id, self.ee_link_index, computeForwardKinematics=True, computeLinkVelocity=1)
            ee_pos = np.array(ee_state[0], dtype=np.float32)
            ee_vel = np.array(ee_state[6], dtype=np.float32) # Vitesse linéaire

            # Position du cube CIBLE
            target_cube_id = self.cube_ids[self.target_cube_index]
            target_cube_pos, _ = p.getBasePositionAndOrientation(target_cube_id)
            target_cube_pos = np.array(target_cube_pos, dtype=np.float32)

            # Position relative EE -> CIBLE
            relative_pos_to_target = target_cube_pos - ee_pos

            # Index du cube cible (normalisé entre 0 et 1 si num_cubes > 1)
            norm_target_index = 0.0
            if self.num_cubes > 1:
                 norm_target_index = float(self.target_cube_index) / (self.num_cubes - 1)

            # Concaténation de l'observation
            obs = np.concatenate([
                joint_pos, joint_vel,
                target_cube_pos, relative_pos_to_target, ee_vel,
                np.array([norm_target_index], dtype=np.float32)
            ])

            # Vérifier si l'observation correspond à l'espace défini
            if not self.observation_space.contains(obs):
                # Cliper ou juste avertir ? Cliper est souvent plus sûr pour SB3.
                # print("Avertissement: Observation en dehors des limites définies. Clipping.")
                obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

            return obs.astype(self.observation_space.dtype) # Assurer le bon type

        except p.error as e:
            print(f"Erreur PyBullet dans _get_obs: {e}. Retour d'une observation nulle.")
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        except IndexError as e:
             print(f"Erreur d'index dans _get_obs (probablement lié à target_cube_index): {e}. Retour d'une obs nulle.")
             return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)


    def _get_info(self) -> Dict[str, Any]:
         """Récupère des informations de débogage/analyse."""
         # Vérifier si l'environnement est prêt
         if self.kuka_id is None or self.target_cube_index < 0 or self.target_cube_index >= len(self.cube_ids):
             return {"status": "invalid_state", "message": "Info non disponible avant reset ou index cible invalide."}

         info = {
             "distance_to_target": np.inf,
             "target_cube_pos": np.zeros(3),
             "target_cube_index": self.target_cube_index,
             "ee_pos": np.zeros(3),
             "ee_vel": np.zeros(3),
             "non_target_distances": [],
             "non_target_displacements": [],
             "num_cubes": self.num_cubes,
             "status": "ok" # Statut par défaut
         }

         try:
             # EE state
             ee_state = p.getLinkState(self.kuka_id, self.ee_link_index, computeForwardKinematics=True, computeLinkVelocity=1)
             ee_pos = np.array(ee_state[0])
             ee_vel = np.array(ee_state[6])
             info["ee_pos"] = ee_pos
             info["ee_vel"] = ee_vel

             # Target cube state
             target_cube_id = self.cube_ids[self.target_cube_index]
             target_cube_pos, _ = p.getBasePositionAndOrientation(target_cube_id)
             target_cube_pos = np.array(target_cube_pos)
             info["target_cube_pos"] = target_cube_pos

             distance_to_target = np.linalg.norm(target_cube_pos - ee_pos)
             info["distance_to_target"] = distance_to_target

             # Infos sur les cubes non-cibles
             for i, cube_id in enumerate(self.cube_ids):
                 if i == self.target_cube_index:
                     continue
                 # Vérifier que l'index i est valide pour cube_ref_positions
                 if i >= len(self.cube_ref_positions):
                      print(f"Avertissement: Index {i} hors limites pour cube_ref_positions dans _get_info.")
                      continue

                 nt_pos, _ = p.getBasePositionAndOrientation(cube_id)
                 nt_pos = np.array(nt_pos)
                 info["non_target_distances"].append(np.linalg.norm(ee_pos - nt_pos))

                 # Comparer à la position de référence
                 ref_pos = self.cube_ref_positions[i]
                 displacement = np.linalg.norm(nt_pos - ref_pos)
                 info["non_target_displacements"].append(displacement)

         except p.error as e:
             info["status"] = "pybullet_error"
             info["message"] = f"Erreur PyBullet dans _get_info: {e}"
             # Réinitialiser les valeurs potentiellement partielles
             info["distance_to_target"] = np.inf
             info["non_target_distances"] = []
             info["non_target_displacements"] = []
         except IndexError as e:
             info["status"] = "index_error"
             info["message"] = f"Erreur d'index dans _get_info: {e}"
             info["distance_to_target"] = np.inf
             info["non_target_distances"] = []
             info["non_target_displacements"] = []


         return info


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """ Exécute une étape dans l'environnement. """
        self.current_step += 1

        terminated = False
        truncated = False
        reward = 0.0

        try:
            # 1. Appliquer l'action (contrôle en vitesse)
            clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
            target_joint_velocities = self.scale_action(clipped_action)

            p.setJointMotorControlArray(
                bodyUniqueId=self.kuka_id,
                jointIndices=range(self.num_arm_joints),
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=target_joint_velocities,
                forces=[self.joint_force] * self.num_arm_joints
            )

            # 2. Simulation PyBullet
            for _ in range(self.substeps):
                p.stepSimulation()
                if self.render_mode == 'human': time.sleep(self.timeStep)


            # 3. Observation et Informations
            observation = self._get_obs()
            info = self._get_info()


            # --- Calcul de la récompense ---
            distance = info["distance_to_target"]
            ee_pos = info["ee_pos"] # Assuré d'être valide si status=="ok"

            # Pénalité de distance au CUBE CIBLE (dense)
            # Utiliser une fonction qui ne devient pas trop négative (ex: exp(-dist))
            # reward -= distance # Simple pénalité linéaire
            reward = -distance # Initialisation simple

            # Bonus de succès si assez proche du CUBE CIBLE
            if distance < self.success_eps:
                reward += 10.0 # Gros bonus pour succès
                terminated = True
                info["is_success"] = True # Ajouter une info pour le Monitor
                # print(f"SUCCES! Cible {self.target_cube_index} atteinte step {self.current_step}")

            # Shaping "best-so-far" pour le CUBE CIBLE
            IMPROVE_BONUS_FACTOR = 2.0 # Ajuster ce facteur
            if distance < self.best_distance - 1e-5: # Seuil pour éviter bruit numérique
                improvement = self.best_distance - distance
                reward += IMPROVE_BONUS_FACTOR * improvement # Bonus proportionnel à l'amélioration
                self.best_distance = distance


            # --- Pénalités liées aux CUBES NON-CIBLES ---
            total_non_target_move_penalty = 0.0
            total_non_target_zone_penalty = 0.0

            if self.num_cubes > 1:
                # Utiliser les infos pré-calculées si disponibles
                non_target_displacements = info.get("non_target_displacements", [])
                non_target_distances = info.get("non_target_distances", [])

                # Pénalité pour déplacement
                for displacement in non_target_displacements:
                    if displacement > self.cube_move_threshold:
                        # Pénalité qui augmente avec le déplacement
                        move_penalty = self.cube_move_penalty_coeff * (displacement - self.cube_move_threshold)
                        total_non_target_move_penalty += move_penalty

                # Pénalité pour violation de zone
                for dist_ee_to_nt in non_target_distances:
                     if dist_ee_to_nt < self.non_target_zone_eps:
                        # Pénalité proportionnelle à la pénétration
                        penetration = self.non_target_zone_eps - dist_ee_to_nt
                        zone_penalty = self.non_target_zone_penalty_coeff * penetration
                        total_non_target_zone_penalty += zone_penalty

            reward -= total_non_target_move_penalty
            reward -= total_non_target_zone_penalty

            # Pénalité de collision Robot <-> Sol
            if self.plane_id is not None and self.check_collision(self.kuka_id, self.plane_id):
                reward -= 10.0 # Pénalité fixe importante
                terminated = True
                info["collision_ground"] = True
            
            # Pénalité de collision Robot <-> cubes
            for cube_id in self.cube_ids:
                if self.check_collision(self.kuka_id, cube_id):
                    reward -= 10.0 # Pénalité fixe importante
                    terminated = True
                    info["collision_cubes"] = True
                    

            # Pénalité sur la magnitude de l'action
            if self.action_penalty_coeff > 0:
                 action_magnitude_penalty = self.action_penalty_coeff * np.mean(np.square(clipped_action)) # Moyenne plutôt que somme
                 reward -= action_magnitude_penalty


            # 4. Vérification de fin d'épisode (temps écoulé)
            if self.current_step >= self._max_episode_steps:
                truncated = True # Fin normale par limite de temps

            # Assurer que terminated et truncated ne sont pas True en même temps (convention Gym)
            if terminated:
                truncated = False

            # Ajouter la récompense totale à l'info pour Monitor
            # info["reward"] = reward
            # if terminated: info.setdefault("is_success", False) # Assurer que is_success existe si terminé
            info["reward"] = reward
            info["is_success"] = info.get("is_success", False)



            # Le rendu est géré par la boucle externe
            # if self.render_mode == "human": self.render()

            return observation, reward, terminated, truncated, info

        except p.error as e:
            print(f"Erreur PyBullet fatale pendant step: {e}. Terminaison de l'épisode.")
            observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            reward = -200.0 # Pénalité très élevée pour erreur grave
            terminated = True
            truncated = False
            info = {"status": "fatal_pybullet_error", "message": str(e)}
            return observation, reward, terminated, truncated, info
        except Exception as e: # Capturer d'autres erreurs potentielles
             print(f"Erreur inattendue pendant step: {e}. Terminaison.")
             observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
             reward = -200.0
             terminated = True
             truncated = False
             info = {"status": "unexpected_error", "message": str(e)}
             return observation, reward, terminated, truncated, info


    def check_collision(self, bodyA_id: int, bodyB_id: int) -> bool:
        """Vérifie s'il y a des points de contact entre deux corps."""
        # Vérifier si les IDs sont valides (non None et >= 0)
        if bodyA_id is None or bodyA_id < 0 or bodyB_id is None or bodyB_id < 0:
            # print("Avertissement: Vérification de collision avec ID invalide.")
            return False
        try:
            contact_points = p.getContactPoints(bodyA=bodyA_id, bodyB=bodyB_id)
            return len(contact_points) > 0
        except p.error as e:
            print(f"Erreur PyBullet pendant getContactPoints: {e}")
            return False # Supposer pas de collision en cas d'erreur


    def render(self) -> Optional[np.ndarray]:
        """
        Génère le rendu de l'environnement.
        'human' mode utilise la fenêtre PyBullet GUI.
        'rgb_array' mode retourne une image numpy.
        """
        if self.render_mode == 'rgb_array':
            # Vérifier si la simulation est active et connectée
            if self.physicsClient is None or not p.isConnected(self.physicsClient):
                 # print("Avertissement: Render appelé sans client PyBullet connecté.")
                 return None # Important de retourner None si pas de client

            # Définir la résolution souhaitée pour la vidéo
            width, height = 1280, 720 

            # Définir les paramètres de la caméra
            target_pos = [0.2, 0, 0.2] # Point cible de la caméra
            distance = 1.8            # Distance de la caméra au point cible
            yaw = -90                 # Rotation autour de Z (vue de côté)
            pitch = -30               # Inclinaison haut/bas
            roll = 0                  # Rotation autour de l'axe de vue

            try:
                # --- CORRECTION : Suppression de physicsClientId ---
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=target_pos,
                        distance=distance,
                        yaw=yaw,
                        pitch=pitch,
                        roll=roll,
                        upAxisIndex=2,
                        # physicsClientId=self.physicsClient # Argument incorrect/inutile
                )
                proj_matrix = p.computeProjectionMatrixFOV(
                        fov=60, # Champ de vision
                        aspect=float(width) / height,
                        nearVal=0.1,
                        farVal=10.0,
                        # physicsClientId=self.physicsClient # Argument incorrect/inutile
                )

                # Utiliser ER_TINY_RENDERER pour la compatibilité
                renderer = p.ER_TINY_RENDERER
                (_, _, px, _, _) = p.getCameraImage(
                    width=width, height=height,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix,
                    renderer=renderer,
                    # physicsClientId=self.physicsClient # Argument incorrect/inutile
                )
                # --- FIN CORRECTION ---

                # Remodeler et retourner l'image RGB
                rgb_array = np.array(px, dtype=np.uint8)
                rgb_array = np.reshape(rgb_array, (height, width, 4)) # RGBA de PyBullet
                return rgb_array[:, :, :3] # Retourner uniquement RGB

            except p.error as e:
                 # Rendre l'erreur plus visible dans les logs
                 print("\n" + "="*25 + " ERREUR CRITIQUE RENDER " + "="*25)
                 print(f"| PyBullet a échoué pendant getCameraImage/render : {e}")
                 print("| Vérifiez l'installation, les drivers, ou si une GUI est nécessaire.")
                 print("| Aucune frame vidéo ne sera enregistrée pour cette étape.")
                 print("="*75 + "\n")
                 return None # Retourner None est crucial en cas d'échec

        elif self.render_mode == 'human':
            # Le rendu est géré par p.connect(p.GUI), rien à faire ici
            pass
        return None # Retourner None si render_mode n'est pas géré


    def close(self):
        """Nettoie et ferme l'environnement PyBullet."""
        if self.physicsClient is not None:
            try:
                if p.isConnected(self.physicsClient):
                    p.disconnect(self.physicsClient)
            except p.error as e:
                print(f"Erreur lors de la déconnexion de PyBullet: {e}")
            finally:
                self.physicsClient = None
                self.kuka_id = None
                self.plane_id = None
                self.cube_ids = []


# --- Fonctions Utilitaires pour l'Entraînement ---

def create_env(
        config: Dict[str, Any], 
        seed: int = 0, 
        log_dir: Optional[str] = None, 
        record_video: bool = False, 
        video_folder: str = 'videos', 
        video_freq: int = 100
    ) -> gym.Env:
    """
    Crée, configure et wrappe une instance de l'environnement.

    Args:
        config: Dictionnaire contenant les paramètres pour KukaMultiCubeReachEnv.
        seed: Graine aléatoire pour l'environnement.
        log_dir: Répertoire pour sauvegarder les logs Monitor. Si None, pas de Monitor.
        record_video: Si True, enregistre des vidéos.
        video_folder: Dossier pour sauvegarder les vidéos.
        video_freq: Fréquence d'enregistrement des vidéos (tous les N épisodes).

    Returns:
        Instance de l'environnement, potentiellement wrappée.
    """
    render_mode = config.get('render_mode', 'rgb_array' if record_video else None)
    env_kwargs = config.copy()
    if record_video : 
        print('okkkkkkkkk')
        env_kwargs['render_mode'] = 'rgb_array'
    else :
        env_kwargs['render_mode'] = render_mode
    print("render_mode",render_mode)
    env = KukaMultiCubeReachEnv(**env_kwargs)
    # Important: Appeler reset avec la seed pour initialiser correctement la graine interne
    env.reset(seed=seed)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        monitor_path = os.path.join(log_dir, "monitor.csv")
        # Supprimer l'ancien fichier monitor s'il existe pour éviter la concaténation
        if os.path.exists(monitor_path):
            try: os.remove(monitor_path)
            except OSError as e: print(f"Avertissement: Impossible de supprimer {monitor_path}: {e}")
        # Clés d'info à logger par Monitor (optionnel mais utile)
        info_keywords = ("is_success",) # Ajouter d'autres clés de 'info' si besoin
        env = Monitor(env, filename=monitor_path, info_keywords=info_keywords)

    if record_video:
        if env_kwargs['render_mode'] != "rgb_array":
            print(f"Avertissement: record_video=True nécessite render_mode='rgb_array'. Reçu: {render_mode}. Pas d'enregistrement vidéo.")
        else:
            print(f"Enregistrement vidéo activé (tous les {video_freq} épisodes) dans {video_folder}")
            env = RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda ep_id: ep_id % video_freq == 0,
            )

    return env


# --- Fonction d'Entraînement ---

def train(
        config: Dict[str, Any]
    ):
    """
    Configure et lance une session d'entraînement Stable Baselines3 PPO.

    Args:
        config: Dictionnaire de configuration contenant:
            - env_config: Paramètres pour KukaMultiCubeReachEnv.
            - train_config: Paramètres pour l'entraînement (total_timesteps, etc.).
            - log_config: Chemins pour logs, modèles, vidéos.
            - ppo_config: Hyperparamètres du modèle PPO.
            - load_config: Options pour charger un modèle existant.
    """
    env_config = config['env_config']
    train_config = config['train_config']
    log_config = config['log_config']
    ppo_config = config['ppo_config']
    load_config = config['load_config']

    # Création des répertoires
    os.makedirs(log_config['log_dir'], exist_ok=True)
    os.makedirs(log_config['tensorboard_log'], exist_ok=True)
    os.makedirs(log_config['model_checkpoints_path'], exist_ok=True)
    os.makedirs(log_config['video_folder'], exist_ok=True)

    # --- Création de l'environnement (vectorisé même pour un seul env) ---
    # Utilisation de DummyVecEnv pour compatibilité avec VecNormalize
    # La fonction lambda encapsule la création de l'environnement unique
    vec_env = DummyVecEnv([lambda: create_env(
        config=env_config,
        seed=train_config['base_seed'],
        log_dir=os.path.join(log_config['log_dir'], "monitor_logs"),
        record_video=train_config['record_video'],
        video_folder=log_config['video_folder'],
        video_freq=train_config['video_freq']
    )])

    # --- Normalisation ---
    norm_stats_path = os.path.join(log_config['log_dir'], "vecnormalize.pkl")
    if load_config['load_vecnormalize'] and os.path.exists(norm_stats_path):
        print(f"Chargement des statistiques VecNormalize depuis: {norm_stats_path}")
        vec_env = VecNormalize.load(norm_stats_path, venv=vec_env)
        vec_env.training = True # S'assurer qu'il est en mode entraînement
        vec_env.norm_reward = train_config.get('normalize_reward', False) # Appliquer la config actuelle
    else:
        print("Création de nouvelles statistiques VecNormalize...")
        vec_env = VecNormalize(vec_env,
                             norm_obs=True, 
                             norm_reward=train_config.get('normalize_reward', False), 
                             clip_obs=train_config.get('clip_obs', 10.0),
                             gamma=ppo_config.get('gamma', 0.99)) # Gamma pour calcul reward normalisé

    # --- Callback de sauvegarde ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(train_config['save_freq'], 1), # Fréquence par pas total
        save_path=log_config['model_checkpoints_path'],
        name_prefix=f"{log_config['env_id']}_ckpt",
        save_replay_buffer=False, # Pas utile pour PPO
        save_vecnormalize=True    # IMPORTANT: sauvegarder les stats
    )

    # --- Initialisation ou chargement du Modèle PPO ---
    model_load_path = load_config.get('model_load_path', None)
    load_model = load_config['load_existing_model'] and model_load_path and os.path.exists(model_load_path)

    if load_model:
         print(f"Chargement du modèle PPO depuis: {model_load_path}")
         # Charger les hyperparamètres custom s'ils sont sauvegardés avec le modèle
         # ou les passer explicitement si nécessaire pour la compatibilité.
         model = PPO.load(model_load_path, env=vec_env, tensorboard_log=log_config['tensorboard_log'],
                           # Peut nécessiter de repasser certains hyperparams si changés:
                           # custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
                           )
         # model.set_env(vec_env) # Assurer que l'env est bien lié
         print("Modèle chargé.")
    else:
        print("Initialisation d'un nouveau modèle PPO...")
        model = PPO(
            env=vec_env,
            tensorboard_log=log_config['tensorboard_log'],
            verbose=1,
            **ppo_config # Passer les hyperparamètres PPO
        )
        print("Nouveau modèle créé.")

    # --- Entraînement ---
    print(f"Début de l'entraînement pour {train_config['total_timesteps']} pas...")
    try:
        model.learn(
            total_timesteps=train_config['total_timesteps'],
            callback=checkpoint_callback,
            log_interval=train_config.get('log_interval', 10), # Logguer stats Monitor
            reset_num_timesteps=not load_model # Réinitialiser compteur si nouveau modèle
        )
    except KeyboardInterrupt:
        print("Entraînement interrompu par l'utilisateur.")
    finally:
        # --- Sauvegarde Finale ---
        print("Sauvegarde du modèle final et des statistiques VecNormalize...")
        final_model_path = os.path.join(log_config['log_dir'], log_config['final_model_name'])
        model.save(final_model_path)
        vec_env.save(norm_stats_path) # Sauvegarde finale des stats VecNormalize
        print(f"Modèle final sauvegardé dans: {final_model_path}.zip")
        print(f"Statistiques VecNormalize sauvegardées dans: {norm_stats_path}")

        # --- Fermeture de l'environnement ---
        vec_env.close() # Important pour nettoyer PyBullet et Monitor

    print("Entraînement terminé.")


# --- Fonction de Chargement et Exécution ---


def load_and_run(
        model_path: str, 
        norm_stats_path: Optional[str], 
        config: Dict[str, Any], 
        num_episodes: int = 5, 
        target_indices: Optional[List[int]] = None,
        continue_episodes: bool = False,
    ):
    """
    Charge un modèle PPO entraîné et le fait tourner dans l'environnement.

    Args:
        model_path: Chemin vers le fichier .zip du modèle PPO sauvegardé.
        norm_stats_path: Chemin vers le fichier .pkl des stats VecNormalize.
        config: Dictionnaire de configuration ('env_config', 'log_config').
        num_episodes: Nombre d'épisodes à exécuter.
        target_indices: Liste optionnelle d'indices cibles à utiliser pour
                        chaque épisode en mode 'eval'. Si None ou plus courte
                        que num_episodes, le mode 'train' (cible aléatoire)
                        sera utilisé pour les épisodes restants.
        continue_episodes: Si True, active le mode où chaque épisode commence
                           depuis l'état final du précédent (sauf la cible).
    """

    if not os.path.exists(model_path):
        print(f"Erreur: Fichier modèle non trouvé à {model_path}")
        return

    env_config = config['env_config'].copy()
    env_config['render_mode'] = 'human'

    def make_env():
         env = KukaMultiCubeReachEnv(**env_config)
         return env
    
    from copy import deepcopy
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

    
    class NoAutoResetDummy(DummyVecEnv):
        """
        Identique à DummyVecEnv mais SANS le reset automatique.
        """
        def step_wait(self) -> VecEnvStepReturn:
            for env_idx in range(self.num_envs):
                obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = \
                    self.envs[env_idx].step(self.actions[env_idx])

                self.buf_dones[env_idx] = terminated or truncated
                # On garde l'observation terminale pour info, mais on ne reset PAS :
                if self.buf_dones[env_idx]:
                    self.buf_infos[env_idx]["terminal_observation"] = obs

                self._save_obs(env_idx, obs)

            return (self._obs_from_buf(),
                    np.copy(self.buf_rews),
                    np.copy(self.buf_dones),
                    deepcopy(self.buf_infos))


    run_vec_env = NoAutoResetDummy([make_env])

    if norm_stats_path and os.path.exists(norm_stats_path):
        print(f"Chargement VecNormalize depuis: {norm_stats_path}")
        vec_env_normalized = VecNormalize.load(norm_stats_path, venv=run_vec_env)
        vec_env_normalized.training = False
        vec_env_normalized.norm_reward = False
        run_env = vec_env_normalized
        print("VecNormalize chargé en mode inférence.")
    else:
        if norm_stats_path: print(f"Avertissement: Fichier VecNormalize non trouvé: {norm_stats_path}.")
        print("Exécution sans normalisation VecNormalize (via DummyVecEnv).")
        run_env = run_vec_env

    print(f"Chargement du modèle PPO depuis: {model_path}")
    try:
        model = PPO.load(model_path, device='auto')
        print("Modèle chargé.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        run_env.close()
        return
    
    print(f"Exécution de {num_episodes} épisodes...")
    if continue_episodes:
        print("Mode 'continuation d'épisodes' activé.")

    print(f"Exécution de {num_episodes} épisodes...")
    total_reward = 0
    num_success = 0

    for episode in range(num_episodes):
        # --- Préparer les options pour CE reset ---
        reset_options = {}
        current_target_idx = None
        if target_indices and episode < len(target_indices):
            current_target_idx = target_indices[episode]
            reset_options['mode'] = 'eval'
            reset_options['target_index'] = current_target_idx
            print(f"\n--- Épisode {episode + 1} (Mode: Eval, Cible: {current_target_idx}) ---")
        else:
            reset_options['mode'] = 'train'
            print(f"\n--- Épisode {episode + 1} (Mode: Train, Cible: Aléatoire) ---")

        if continue_episodes:
            reset_options['start_from_previous_state'] = True

        # --- Appel à env_method AVANT reset ---
        try:
            # Appeler la méthode 'set_next_reset_options' sur l'env 0 avec les 'reset_options'
            run_env.env_method("set_next_reset_options", reset_options, indices=[0])

            # Appeler reset sur le VecEnv SANS l'argument 'options'
            print("DEBUG: Appelant run_env.reset() sans appel env_method préalable immédiat.")
            obs = run_env.reset()
  

        except (ValueError, RuntimeError, AttributeError) as e: # Capturer erreurs potentielles
             print(f"Erreur lors du reset (via env_method) pour l'épisode {episode + 1}: {e}")
             print("Passage à l'épisode suivant.")
             continue # Sauter cet épisode

        # --- Boucle d'exécution de l'épisode ---
        terminated = False
        truncated = False
        ep_reward = 0
        ep_steps = 0
        actual_target_idx = "?"

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            new_obs, rewards, dones, infos = run_env.step(action)

            obs = new_obs
            reward = rewards[0]
            terminated = dones[0]
            info = infos[0]
            truncated = info.get('TimeLimit.truncated', False)
            if terminated:
                truncated = False

            ep_reward += reward
            ep_steps += 1

            if ep_steps == 1:
                actual_target_idx = info.get('target_cube_index', '?')
                if reset_options.get('mode') == 'train': # Vérifier mode réel utilisé
                     print(f"   (Cible choisie aléatoirement : {actual_target_idx})")

            time.sleep(0.01)

            if terminated and info.get("is_success", False):
                num_success += 1

        # --- Fin de l'épisode ---
        final_info = info
        print(f"Épisode {episode + 1}: Terminé après {ep_steps} pas. Récompense={ep_reward:.2f}. "
              f"Terminated={terminated}, Truncated={truncated}. Succès: {final_info.get('is_success', False)}")
        total_reward += ep_reward

    # --- Fin de l'exécution ---
    print("\n--- Résultats d'exécution ---")
    print(f"Récompense moyenne sur {num_episodes} épisodes: {total_reward / num_episodes:.2f}")
    print(f"Taux de succès: {num_success / num_episodes * 100:.1f}%")

    run_env.close()
    print("Exécution terminée.")



# --- Configuration Principale et Exécution ---
if __name__ == "__main__":
    import argparse

    # Configuration par défaut (modifiable via CLI ou directement ici)
    CONFIG = {
        "env_config": {
            "render_mode": None, # Sera 'rgb_array' si record_video=True
            "num_cubes": DEFAULT_NUM_CUBES,
            "cube_size": 0.04,
            "r_min": 0.40,
            "r_max": 0.82,
            "max_episode_steps": 20,
            "substeps": DEFAULT_SUBSTEPS,
            "success_eps": DEFAULT_SUCCESS_EPS,
            "ee_link_index": DEFAULT_EE_LINK_INDEX,
            "joint_force": 100.0,
            "cube_move_penalty_coeff": 5.0,
            "cube_move_threshold": 0.01,
            "non_target_zone_penalty_coeff": 2.0,
            "non_target_zone_eps": DEFAULT_NON_TARGET_ZONE_EPS,
            "action_penalty_coeff": 0.001,
            "max_ee_velocity": 2.0,
            "min_dist_between_cubes": DEFAULT_MIN_DIST_BETWEEN_CUBES,
            "continue_training_episodes": False,
        },
        "train_config": {
            "total_timesteps": 20_000_000, # Réduire pour test rapide ?
            "save_freq": 50_000,
            "base_seed": 42,
            "normalize_reward": False, # Souvent False pour les tâches de Reach
            "clip_obs": 10.0,
            "record_video": True, # Mettre à False pour désactiver l'enregistrement
            "video_freq": 5000,    # Enregistrer tous les 200 épisodes
            "log_interval": 5,   # Fréquence de log des récompenses (en nombre d'épisodes)
        },
        "log_config": {
            "env_id": "KukaMultiCubeReach_v1",
            "log_dir": "logs/ppo_kuka_multi_cube_single",
            "tensorboard_log": "logs/ppo_kuka_multi_cube_single/tensorboard/",
            "model_checkpoints_path": "logs/ppo_kuka_multi_cube_single/checkpoints/",
            "video_folder": "videos/ppo_kuka_multi_cube_single/",
            "final_model_name": "ppo_kuka_multi_cube_final.zip"
        },
        "ppo_config": {
            # Hyperparamètres PPO (valeurs par défaut SB3 )
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,        # Steps par collecte (par environnement, ici 1 seul)
            "batch_size": 64,       # Taille du minibatch pour l'optimisation
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,        # Coeff d'entropie (peut aider exploration)
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "device": "auto",       # Utilise GPU si disponible
            "policy_kwargs": dict(net_arch=dict(pi=[64, 64], vf=[64, 64])) # Petite architecture pour commencer
        },
        "load_config": {
            "load_existing_model": False, # Mettre à True pour charger depuis model_load_path
            "model_load_path": "logs/ppo_kuka_multi_cube_single/checkpoints/KukaMultiCubeReach_v1_ckpt_XXXXXX_steps.zip", # Remplacer XXXXXX
            "load_vecnormalize": True, # Essayer de charger les stats si on charge un modèle
        }
    }

    # Analyse des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Entraîner ou exécuter un agent PPO pour KukaMultiCubeReach.")
    parser.add_argument("--train", action="store_true", help="Lancer une nouvelle session d'entraînement.")
    parser.add_argument("--load", type=str, metavar="MODEL_PATH", help="Charger un modèle depuis le chemin spécifié et l'exécuter.")
    parser.add_argument("--timesteps", type=int, default=CONFIG['train_config']['total_timesteps'], help="Nombre total de pas pour l'entraînement.")
    parser.add_argument("--episodes", type=int, default=10, help="Nombre d'épisodes à exécuter en mode --load.")
    parser.add_argument("--logdir", type=str, default=CONFIG['log_config']['log_dir'], help="Répertoire racine pour les logs, modèles et vidéos.")
    parser.add_argument("--cubes", type=int, default=CONFIG['env_config']['num_cubes'], help="Nombre de cubes dans l'environnement.")
    parser.add_argument("--seed", type=int, default=CONFIG['train_config']['base_seed'], help="Graine aléatoire de base.")
    
    parser.add_argument("--targets", type=int, nargs='+', 
                        help="Liste d'indices cibles à utiliser en mode --load (ex: --targets 0 2 1). Active le mode 'eval' pour ces épisodes.")
    
    parser.add_argument("--record", action="store_true", # Ne pas mettre de default ici, action="store_true" suffit
                        help="Activer l'enregistrement vidéo pendant l'entraînement (défaut: inactif).")
    
    parser.add_argument("--video-freq", type=int, default=CONFIG['train_config']['video_freq'],
                        help="Fréquence d'enregistrement vidéo (tous les N épisodes).")
    
    parser.add_argument("--continue-episodes", action="store_true",
                        help="En mode --load, enchaîner les épisodes sans réinitialiser l'état (sauf la cible).")


    args = parser.parse_args()

    # Mettre à jour la configuration avec les arguments CLI
    CONFIG['train_config']['total_timesteps'] = args.timesteps
    CONFIG['train_config']['base_seed'] = args.seed
    CONFIG['env_config']['num_cubes'] = args.cubes
    CONFIG['log_config']['log_dir'] = args.logdir # Important de le faire en premier

    # Mettre à jour les autres chemins basés sur le logdir final
    log_dir = CONFIG['log_config']['log_dir'] # Utiliser la valeur finale
    CONFIG['log_config']['tensorboard_log'] = os.path.join(log_dir, "tensorboard/")
    CONFIG['log_config']['model_checkpoints_path'] = os.path.join(log_dir, "checkpoints/")
    CONFIG['log_config']['video_folder'] = os.path.join(log_dir, "videos/")
    CONFIG['log_config']['final_model_name'] = f"{CONFIG['log_config']['env_id']}_final.zip"
    default_load_path = os.path.join(log_dir, CONFIG['log_config']['final_model_name'])
    # Mettre à jour model_load_path seulement si --load n'a pas été utilisé
    if not args.load:
         CONFIG['load_config']['model_load_path'] = default_load_path

    # Mettre à jour la config vidéo
    CONFIG['train_config']['record_video'] = args.record
    CONFIG['train_config']['video_freq'] = args.video_freq # Sera la val

    CONFIG['env_config']['continue_training_episodes'] = args.continue_episodes

    model_to_load = args.load if args.load else None


    if args.train:
        print("--- Démarrage de l'entraînement ---")
        load_path_for_train = CONFIG['load_config']['model_load_path']
        # Charger un modèle existant SEULEMENT s'il existe ET que l'utilisateur ne force PAS un nouvel entraînement
        # (On pourrait ajouter un flag --force-new ou considérer que --train seul implique nouveau)
        # Logique actuelle: si le modèle final par défaut existe, on continue.
        CONFIG['load_config']['load_existing_model'] = os.path.exists(load_path_for_train)
        if CONFIG['load_config']['load_existing_model']:
             print(f"Chargement depuis {load_path_for_train} pour continuer.")
        else:
             print(f"Aucun modèle trouvé à {load_path_for_train}. Nouvel entraînement.")
             CONFIG['load_config']['load_vecnormalize'] = False # Ne pas charger si nouveau

        # Passer la config mise à jour
        train(CONFIG)

    elif model_to_load: # Utiliser la variable définie plus haut
        print(f"--- Chargement et Exécution du modèle: {model_to_load} ---")
        # Essayer de trouver vecnormalize.pkl relative au modèle chargé
        norm_stats_path = os.path.join(os.path.dirname(model_to_load), "vecnormalize.pkl")
        if not os.path.exists(norm_stats_path):
            norm_stats_path_alt = os.path.join(os.path.dirname(os.path.dirname(model_to_load)), "vecnormalize.pkl")
            if os.path.exists(norm_stats_path_alt):
                norm_stats_path = norm_stats_path_alt
            else:
                norm_stats_path = None
        load_and_run(
            model_path=model_to_load,
            norm_stats_path=norm_stats_path,
            config=CONFIG, # Passer la config mise à jour
            num_episodes=args.episodes,
            target_indices=args.targets,
            continue_episodes=args.continue_episodes
            )
    else:
        print("Veuillez spécifier une action: --train ou --load <chemin_modele>")
        parser.print_help()

    print("--- Script terminé ---")