import dataclasses
import operator
import pytest
import jax
import pyqg_jax


@pytest.mark.parametrize(
    "attrs", [(), ("state",), ("t",), ("tc",), ("state", "t", "tc")]
)
def test_replace_updates(attrs):
    base_state = pyqg_jax.steppers.StepperState(
        state=pyqg_jax.qg_model.QGModel().create_initial_state(jax.random.key(0)),
        t=1.0,
        tc=2,
    )
    replacements = {}
    for attr in attrs:
        if attr in {"t", "tc"}:
            replacement = getattr(base_state, attr) + 1
        else:
            leaves, treedef = jax.tree_util.tree_flatten(getattr(base_state, attr))
            replacement = jax.tree_util.tree_unflatten(treedef, leaves)
        replacements[attr] = replacement
    new_state = base_state.update(**replacements)
    assert new_state is not base_state
    for field in map(
        operator.attrgetter("name"), dataclasses.fields(pyqg_jax.steppers.StepperState)
    ):
        new_attr = getattr(new_state, field)
        old_attr = getattr(base_state, field)
        if field not in attrs:
            assert new_attr is old_attr
        else:
            assert new_attr is not old_attr


@pytest.mark.parametrize("extra_args", [("argx",), ("argx", "argy")])
def test_update_extra_args(extra_args):
    base_state = pyqg_jax.steppers.StepperState(
        state=pyqg_jax.qg_model.QGModel().create_initial_state(jax.random.key(0)),
        t=1.0,
        tc=2,
    )
    replace_args = {}
    for attr in extra_args:
        replace_args[attr] = 1
    with pytest.raises(ValueError, match="invalid state updates") as exc_info:
        _ = base_state.update(tc=3, **replace_args)
    msg = exc_info.value.args[0]
    for attr in extra_args:
        assert attr in msg
