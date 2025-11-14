# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .actor_critic_with_estimator import ActorCriticRecurrentAndEstimator
from .actor_critic_mlp_with_estimator import ActorCriticMlpAndEstimator
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent
from .cpg import CPG_RL
from .actor_critic_roa import ActorCriticROA
from .actor_critic_rma import ActorCriticRMA
from .actor_critic_vae import ActorCriticVAE

__all__ = ["ActorCritic", "ActorCriticRecurrent", "EmpiricalNormalization",
           "RandomNetworkDistillation", "ActorCriticRecurrentAndEstimator",
           "ActorCriticMlpAndEstimator","StudentTeacher","StudentTeacherRecurrent","CPG_RL",
           "ActorCriticROA", "ActorCriticRMA", "ActorCriticVAE"]
