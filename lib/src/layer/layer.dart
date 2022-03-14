import 'package:grizzly_ann/grizzly_ann.dart';

abstract class Layer<InputType, OutputType> {
  ActivationFunction get activationFunction;

  dynamic get weights;
  dynamic get bias;

  OutputType compute(InputType input,
      {void Function(OutputType a) recordActivation});

  UpdateResult calculateErrorLastLayer(OutputType y, OutputType yHat,
      Iterable<num> a, LossFunction lossFunction);

  UpdateResult calculateError(Iterable<num> a, Iterable<num> error);

  void updateWeights(
      Iterable<num> input, Iterable<num> error, double learningRate);
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

  @override
  final List<List<double>> weights;

  final List<double> bias;

  final bool useBias;

  Dense(this.inputSize, this.outputSize,
      {this.activationFunction = ActivationFunction.identity,
      this.useBias = true})
      : weights = MatrixMaker.filled(inputSize, outputSize, 0.0),
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

    final propagatedError = weights.matmultColVector(error);

    return UpdateResult(error, propagatedError);
  }

  @override
  UpdateResult calculateError(
      Iterable<num> a, Iterable<num> prevPropagatedError) {
    final error = prevPropagatedError * a.map(activationFunction.derivative);
    final propagatedError = weights.matmultColVector(error);
    return UpdateResult(error, propagatedError);
  }

  @override
  void updateWeights(
      Iterable<num> input, Iterable<num> error, double learningRate) {
    final weightDelta = input.matmultRowVector(error);
    weights.assignAddition(weightDelta * learningRate);
    bias.assignAddition(error * learningRate);
  }
}

class UpdateResult {
  final error;

  final propagatedError;

  UpdateResult(this.error, this.propagatedError);
}
