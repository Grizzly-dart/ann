import 'package:ann/ann.dart';
import 'package:grizzly_array/grizzly_array.dart';

abstract class LossFunction {
  String get name;

  double compute(Iterable<num> y, Iterable<double> yDash);

  Double1D derivative(Iterable<num> y, Iterable<double> yHat,
      Iterable<double> x, ActivationFunction activationFunction);

  static const meanSquaredError = MeanSquaredErrorLossFunction();

  static const meanAbsoluteError = MeanAbsoluteErrorLossFunction();
}

class MeanSquaredErrorLossFunction implements LossFunction {
  const MeanSquaredErrorLossFunction();

  @override
  final String name = 'Mean squared error';

  @override
  double compute(Iterable<num> y, Iterable<double> yHat) {
    return ((Double1D.own(yHat) - Double1D.own(y))..squareSelf()).mean;
  }

  Double1D derivative(Iterable<num> y, Iterable<double> yHat,
      Iterable<double> x, ActivationFunction activationFunction) {
    return (Double1DView.own(yHat) - Double1DView.own(y))
      ..dot(x.map((e) => activationFunction.derivative(e)));
  }
}

class MeanAbsoluteErrorLossFunction implements LossFunction {
  const MeanAbsoluteErrorLossFunction();

  @override
  final String name = 'Mean squared error';

  @override
  double compute(Iterable<num> y, Iterable<double> yDash) {
    return ((Double1D.own(y) - Double1D.own(yDash))..abs()).mean;
  }

  Double1D derivative(Iterable<num> y, Iterable<double> yDash,
      Iterable<double> x, ActivationFunction activationFunction) {
    throw UnimplementedError();
  }
}
