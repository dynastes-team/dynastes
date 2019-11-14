import tensorflow as tf
def custom_gradient_back_fn(back_f):
    def _recompute_grad_custom_back_fn(f):
      """An eager-compatible version of recompute_grad.
      For f(*args, **kwargs), this supports gradients with respect to args, or to
      gradients with respect to any variables residing in the kwarg 'variables'.
      Note that for keras layer and model objects, this is handled automatically.
      Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
      be able to access the member variables of that object, because `g` returns
      through the wrapper function `inner`.  When recomputing gradients through
      objects that inherit from keras, we suggest keeping a reference to the
      underlying object around for the purpose of accessing these variables.
      Args:
        f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.
      Returns:
       A function `g` that wraps `f`, but which recomputes `f` on the backwards
       pass of a gradient call.
      """
      # TODO(cdfreeman) Add is_recomputing functionality from graph mode version

      @tf.custom_gradient
      def inner(*args, **kwargs):
        """Inner function closure for calculating gradients."""
        result = f(*args, **kwargs)

        def grad(dresult, variables=None):
          """Gradient function calculation for inner function."""
          with tf.GradientTape() as t:
            t.watch(args)
            if variables is not None:
              t.watch(variables)
            with tf.control_dependencies([dresult]):
              result = back_f(*args, **kwargs)
          kw_vars = []
          if variables is not None:
            kw_vars = list(variables)
          grads = t.gradient(
              result, list(args) + kw_vars, output_gradients=[dresult])
          return grads[:len(args)], grads[len(args):]

        return result, grad

      return inner
    return _recompute_grad_custom_back_fn
