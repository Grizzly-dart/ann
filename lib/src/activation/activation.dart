import 'dart:math' as math;

abstract class ActivationFunction {
  String get name;

  double compute(num x);

  double derivative(num y);

  static const identity = LinearActivation();

  static ActivationFunction linear({double a = 1}) => LinearActivation(a: a);

  static const sigmoid = SigmoidActivation();
}

class LinearActivation implements ActivationFunction {
  final double a;

  const LinearActivation({this.a = 1});

  @override
  final name = 'Linear';

  @override
  double compute(num x) => a * x;

  @override
  double derivative(num y) => a;
}


class ReLUActivation implements ActivationFunction {
  const ReLUActivation();

  @override
  final name = 'ReLU';

  @override
  double compute(num x) => math.max(x.toDouble(), 0);

  @override
  double derivative(num y) => y > 0 ? 1 : 0;
}


class PReLUActivation implements ActivationFunction {
  /// Slope of negative section.
  final double alpha;

  const PReLUActivation(this.alpha);

  @override
  final name = 'PReLU';

  @override
  double compute(num x) => x < 0 ? alpha * x : x.toDouble();

  @override
  double derivative(num y) => y < 0 ? alpha : 1;
}

/// Exponential Linear Unit activation function.
/// The hyperparameter [alpha] controls the degree to which ELU saturated for
/// negative input values.
///
/// ELUs diminish the vanishing gradient effect.
///
/// ELUs have negative values which pushes the mean of the activations closer to zero.
/// Mean activations that are closer to zero enable faster learning as they
/// bring the gradient closer to the natural gradient.
/// ELUs saturate to a negative value when the argument gets smaller.
/// Saturation means a small derivative which decreases the variation
/// and the information that is propagated to the next layer.
class ELUActivation implements ActivationFunction {
  /// Slope of negative section. [alpha] controls the degree to which ELU saturated
  /// for negative input values.
  final double alpha;

  ELUActivation({this.alpha = 1.0});

  @override
  final name = 'ELU';

  @override
  double compute(num x) => x < 0 ? alpha * (math.exp(x) - 1) : x.toDouble();

  @override
  double derivative(num y) => y < 0 ? compute(y) + alpha : 1;
}

/// Sigmoid activation function.
///
/// For small values (<-5), it returns a value close to zero, and for large
/// values (>5) the result of the function gets close to 1.
///
/// Sigmoid is equivalent to a 2-element Softmax, where the second element is
/// assumed to be zero. The sigmoid function always returns a value between
/// 0 and 1.
class SigmoidActivation implements ActivationFunction {
  const SigmoidActivation();

  @override
  final name = 'Sigmoid';

  @override
  double compute(num x) => 1 / (1 + math.exp(-x));

  @override
  double derivative(num y) {
    final sig = compute(y);
    return sig * (1 - sig);
  }
}

/// Hyperbolic tangent activation function.
class TanhActivation implements ActivationFunction {
  const TanhActivation();

  @override
  final name = 'tanh';

  @override
  double compute(num x) => (1 - math.exp(-2 * x)) / (1 + math.exp(-2 * x));

  @override
  double derivative(num y) => 1.0 - math.pow(y, 2);
}

class SoftPlusActivation implements ActivationFunction {
  const SoftPlusActivation();

  @override
  final name = 'SoftPlus';

  @override
  double compute(num x) => math.log(1 + math.exp(x));

  @override
  double derivative(num y) => 1 / (1 + math.exp(-y));
}

/// Softmax activation function converts values into a probability distribution.
/// The returned outputs are in range (0, 1).
/* TODO figure out how to use
class SoftmaxActivation implements ActivationFunction {
  const SoftmaxActivation();

  @override
  final name = 'Softmax';

  @override
  double compute(num x) {
    // TODO
    throw UnimplementedError();
  }

  @override
  double derivative(num y) {
    // TODO
    throw UnimplementedError();
  }
}
 */
