import 'package:ann/ann.dart';
import 'package:ann/src/activation/activation.dart';
import 'package:ann/src/loss_function/loss_function.dart';
import 'package:grizzly_array/grizzly_array.dart';

abstract class Layer<InputType, OutputType> {
  OutputType compute(InputType input,
      {void Function(OutputType a) recordActivation});

  UpdateResult calculateErrorLastLayer(OutputType y, OutputType yHat,
      Iterable<num> a, LossFunction lossFunction);

  UpdateResult calculateError(Iterable<num> a, Iterable<double> error);

  void updateWeights(
      Iterable<double> input, Iterable<double> error, double learningRate);
}

abstract class Layer1D implements Layer<Iterable<num>, Double1D> {
  int get inputSize;

  int get outputSize;

  Double1D compute(Iterable<num> input,
      {void Function(Double1D a) recordActivation});
}

class Dense implements Layer1D {
  final int inputSize;

  final int outputSize;

  final ActivationFunction activationFunction;

  final Double2D weights;

  final Double1D bias;

  final bool useBias;

  Dense(this.inputSize, this.outputSize,
      {this.activationFunction = ActivationFunction.sigmoid,
      this.useBias = true})
      : weights = Double2D.sized(inputSize, outputSize),
        bias = Double1D.sized(outputSize);

  @override
  Double1D compute(Iterable<num> input,
      {void Function(Double1D a) recordActivation}) {
    if (input.length != inputSize) {
      throw Exception('invalid input dimension');
    }

    final Double1D out = Double1DView.own(input).matmul(weights)[0].clone();

    if (useBias) {
      out.addition(bias);
    }

    if (recordActivation != null) recordActivation(out.clone());

    if (activationFunction != null) {
      out.apply(activationFunction.compute);
    }

    return out;
  }

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
