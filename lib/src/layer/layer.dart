import 'package:grizzly/grizzly.dart';
import 'package:grizzly_ann/grizzly_ann.dart';
// TODO import 'package:grizzly_array/grizzly_array.dart';

abstract class Layer<InputType, OutputType> {
  OutputType compute(InputType input,
      {void Function(OutputType a) recordActivation});

  UpdateResult calculateErrorLastLayer(OutputType y, OutputType yHat,
      Iterable<num> a, LossFunction lossFunction);

  UpdateResult calculateError(Iterable<num> a, Iterable<double> error);

  void updateWeights(
      Iterable<double> input, Iterable<double> error, double learningRate);
}

abstract class Layer1D implements Layer<Iterable<num>, List<double>> {
  int get inputSize;

  int get outputSize;

  @override
  List<double> compute(Iterable<num> input,
      {void Function(List<double> a) recordActivation});
}

class Dense implements Layer1D {
  @override
  final int inputSize;

  @override
  final int outputSize;

  final ActivationFunction activationFunction;

  final List<List<double>> weights;

  final List<double> bias;

  final bool useBias;

  Dense(this.inputSize, this.outputSize,
      {this.activationFunction = ActivationFunction.sigmoid,
      this.useBias = true})
      : weights = Double2D.sized(inputSize, outputSize),
        bias = List<double>.filled(outputSize, 0);

  @override
  List<double> compute(Iterable<num> input,
      {void Function(List<double> a)? recordActivation}) {
    if (input.length != inputSize) {
      throw Exception('invalid input dimension');
    }

    final List<double> out = input.matmult(weights).cast<double>().toList();

    if (useBias) {
      out.assignAddition(bias);
    }

    if (recordActivation != null) recordActivation(List.from(out));

    out.apply(activationFunction.compute);

    return out;
  }

  @override
  UpdateResult calculateErrorLastLayer(Iterable<double> y,
      Iterable<double> yHat, Iterable<num> a, LossFunction lossFunction) {
    final error = lossFunction.derivative(y, yHat, a, activationFunction);

    final propagatedError = weights.mapTo1D((Double1DView e) => e.dot(error));

    return UpdateResult(error, propagatedError);
  }

  @override
  UpdateResult calculateError(
      Iterable<num> a, Iterable<double> prevPropagatedError) {
    final error = Double1DView.own(prevPropagatedError) *
        a.map(activationFunction.derivative);
    final propagatedError = weights.mapTo1D((Double1DView e) => e.dot(error));
    return UpdateResult(error, propagatedError);
  }

  @override
  void updateWeights(
      Iterable<double> input, Iterable<double> error, double learningRate) {
    final weightDelta = Double1DView.own(input).matmulRow(error);
    weights.subtract(weightDelta * learningRate);
    bias.applyIndexed(
        (value, index) => value - (learningRate * error.elementAt(index)));
  }
}

class UpdateResult {
  final error;

  final propagatedError;

  UpdateResult(this.error, this.propagatedError);
}
