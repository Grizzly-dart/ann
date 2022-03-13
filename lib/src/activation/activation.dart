import 'dart:math' as math;

abstract class ActivationFunction {
  String get name;

  double compute(num input);

  double derivative(num input);
  
  static const linear = LinearActivation();

  static const sigmoid = SigmoidActivation();
}

class LinearActivation implements ActivationFunction {
  const LinearActivation();

  @override
  final name = 'linear';

  @override
  double compute(num input) {
    return input.toDouble();
  }

  @override
  double derivative(num input) => 1;
}

class SigmoidActivation implements ActivationFunction {
  const SigmoidActivation();

  @override
  final name = 'sigmoid';

  @override
  double compute(num input) {
    return 1 / (1 + math.exp(-input));
  }

  @override
  double derivative(num input) {
    final sig = compute(input);
    return sig * (1 - sig);
  }
}
