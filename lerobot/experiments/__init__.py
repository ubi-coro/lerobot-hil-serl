from lerobot.experiments.deprecated.reach_pose import SACRealReachPoseConfig, RealReachPoseEnvConfig
from lerobot.experiments.deprecated.reach_pose_sparse import SACRealReachPoseSparseConfig, RealReachPoseSparseEnvConfig
from lerobot.experiments.deprecated.grasp_cube import SACRealGraspCubeConfig, RealGraspCubeEnvConfig
from lerobot.experiments.deprecated.nist_insertion import (
    # environments
    UR3_NIST_Insertion_XYC_Small,
    UR3_NIST_Insertion_XY_Small,
    UR3_NIST_Insertion_XYC_Medium,
    UR3_NIST_Insertion_XY_Medium,
    UR3_NIST_Insertion_XYC_Large,
    UR3_NIST_Insertion_XY_Large,

    # policies
    SAC_NIST_Insertion_XYC,
    SAC_NIST_Insertion_XY,
    DAgger_NIST_Insertion_XYC
)
from lerobot.experiments.han_insertion.rlpd_sparse import (
    HAN_Insertion_RLPD_Sparse,
    HAN_Insertion_RLPD_Sparse_NoVision,
    HAN_Insertion_RLPD_Sparse_DGN
)
from lerobot.experiments.han_insertion.rlpd_dense import (
    HAN_Insertion_RLPD_Dense,
    HAN_Insertion_RLPD_Dense_NoPriors,
    HAN_Insertion_RLPD_Dense_DGN
)
from lerobot.experiments.han_insertion.bc_dagger import HAN_Insertion_BC_DAgger
from lerobot.experiments.han_insertion.eval import (
    HAN_Insertion_Record_Forces,
    HAN_Insertion_Static_Limits,
    HAN_Insertion_Adaptive_Limits,
    HAN_Insertion_Random_Policy
)