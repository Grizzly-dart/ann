import 'package:grizzly_ann/grizzly_ann.dart';

export 'sequential.dart';

abstract class Network<InputType, OutputType> {
  Iterable<Layer> get layers;

  void addLayer(Layer layer);

  void train(InputType x, OutputType y);

  void trainBatch(Iterable<InputType> x, Iterable<OutputType> y);

  OutputType predict(InputType x);
}