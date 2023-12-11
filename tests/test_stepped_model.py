import jax
import pyqg_jax


def test_tree_flatten_roundtrip():
    stepped_model = pyqg_jax.steppers.SteppedModel(
        pyqg_jax.qg_model.QGModel(),
        pyqg_jax.steppers.EulerStepper(1.0),
    )
    leaves, treedef = jax.tree_util.tree_flatten(
        stepped_model,
        is_leaf=lambda obj: not isinstance(obj, pyqg_jax.steppers.SteppedModel),
    )
    restored_model = jax.tree_util.tree_unflatten(treedef, leaves)
    assert vars(restored_model) == vars(stepped_model)
