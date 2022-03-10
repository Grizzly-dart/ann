import 'dart:math' as math;

abstract class ActivationFunction {
  String get name;

  double compute(num input);

  double derivative(num input);

  static const sigmoid = Sigmoid();
}

class Sigmoid implements ActivationFunction {
  const Sigmoid();

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
