from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame


def test_policy_action_dim_differential_axes_are_scalar():
    frame = TaskFrame(
        policy_mode=[PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE, None, None, None, None],
        control_mode=[ControlMode.VEL, ControlMode.FORCE, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
    )
    assert frame.policy_action_dim == 2


def test_policy_action_dim_absolute_rotation_uses_manifold_dims():
    frame_one = TaskFrame(
        policy_mode=[None, None, None, PolicyMode.ABSOLUTE, None, None],
        control_mode=[ControlMode.POS] * 6,
    )
    frame_two = TaskFrame(
        policy_mode=[None, None, None, PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE, None],
        control_mode=[ControlMode.POS] * 6,
    )
    frame_three = TaskFrame(
        policy_mode=[None, None, None, PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE],
        control_mode=[ControlMode.POS] * 6,
    )

    assert frame_one.policy_action_dim == 2
    assert frame_two.policy_action_dim == 3
    assert frame_three.policy_action_dim == 6


def test_policy_action_dim_mixed_translational_relative_and_absolute_rotation():
    frame = TaskFrame(
        policy_mode=[PolicyMode.ABSOLUTE, None, None, PolicyMode.RELATIVE, PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE],
        control_mode=[ControlMode.POS] * 6,
    )

    # +1 for x absolute pos, +1 for rx relative pos, +3 for two absolute rotation axes (S2)
    assert frame.policy_action_dim == 5
