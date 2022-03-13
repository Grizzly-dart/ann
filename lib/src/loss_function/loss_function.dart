import 'package:grizzly/grizzly.dart';
import 'package:grizzly_ann/grizzly_ann.dart';

abstract class LossFunction {
  String get name;

  double compute(Iterable<num> y, Iterable<double> yDash);

  Iterable<double> derivative(Iterable<num> y, Iterable<double> yHat,
      Iterable<num> x, ActivationFunction activationFunction);

  static const meanSquaredError = MeanSquaredErrorLossFunction();

  static const meanAbsoluteError = MeanAbsoluteErrorLossFunction();
}

class MeanSquaredErrorLossFunction implements LossFunction {
  const MeanSquaredErrorLossFunction();

  @override
  final String name = 'Mean squared error';

  @override
  double compute(Iterable<num> y, Iterable<double> yHat) {
    // TODO ((yHat - y) * (yHat - y)).sqrt().mean;
    throw UnimplementedError();
  }

  @override
  Iterable<double> derivative(Iterable<num> y, Iterable<double> yHat,
      Iterable<num> x, ActivationFunction activationFunction) {
    return (y.toDoubles() - yHat) * x.map(activationFunction.derivative);
  }
}

class MeanAbsoluteErrorLossFunction implements LossFunction {
  const MeanAbsoluteErrorLossFunction();

  @override
  final String name = 'Mean squared error';

  @override
  double compute(Iterable<num> y, Iterable<double> yDash) =>
      (yDash - y).abs().mean;

  @override
  Iterable<double> derivative(Iterable<num> y, Iterable<double> yDash,
      Iterable<num> x, ActivationFunction activationFunction) {
    throw UnimplementedError();
  }
}
