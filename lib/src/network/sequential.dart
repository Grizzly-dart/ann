import 'package:grizzly/grizzly.dart';
import 'package:grizzly_ann/grizzly_ann.dart';

class State {
  var input;

  var activation;

  var error;
}

class SequentialNetwork<InputType, OutputType>
    implements Network<InputType, OutputType> {
  final _layers = <Layer>[];

  @override
  List<Layer> get layers => _layers.toList();

  double learningRate; // TODO

  SequentialNetwork({this.learningRate = 1e-3});

  @override
  void addLayer(Layer layer) {
    _layers.add(layer);
  }

  @override
  void train(InputType x, OutputType y,
      {LossFunction lossFunction = LossFunction.meanSquaredError}) {
    dynamic nextInput = x;

    final states = <State>[];

    for (int i = 0; i < _layers.length; i++) {
      Layer layer = _layers[i];
      final layerState = State()..input = nextInput;
      nextInput = layer.compute(nextInput,
          recordActivation: (a) => layerState.activation = a);
      states.add(layerState);
    }

    UpdateResult updateResult = _layers.last.calculateErrorLastLayer(
        y, nextInput, states.last.activation, lossFunction);
    states.last.error = updateResult.error;

    for (int i = _layers.length - 2; i >= 0; i--) {
      Layer layer = _layers[i];
      final layerState = states[i];
      updateResult = layer.calculateError(
          layerState.activation, updateResult.propagatedError);
      layerState.error = updateResult.error;
    }

    for (int i = 0; i < _layers.length; i++) {
      Layer layer = _layers[i];
      final layerState = states[i];
      layer.updateWeights(layerState.input, layerState.error, learningRate);
    }
  }

  @override
  void trainBatch(Iterable<InputType> x, Iterable<OutputType> y) {
    // TODO
  }

  @override
  OutputType predict(InputType x) {
    dynamic nextInput = x;

    for (Layer layer in _layers) {
      nextInput = layer.compute(nextInput);
    }

    return nextInput;
  }
}

class Sequential1DNetwork
    extends SequentialNetwork<Iterable<num>, List<double>> {
  Sequential1DNetwork({double learningRate = 1e-3})
      : super(learningRate: learningRate);
}
