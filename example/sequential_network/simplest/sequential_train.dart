import 'dart:math';

import 'package:grizzly_ann/grizzly_ann.dart';

void main() {
  final rand = Random(5);
  final x1 = rand.ints(1000, max: 100).toList();
  print(x1);

  final y = (x1 * 5 + 2).toDoubles().toList();

  final inputs = MatrixMaker.fromColumn(x1);
  print(inputs);
  print(y);

  final network = Sequential1DNetwork(learningRate: 1e-5);

  final layer1 = Dense(1, 1);
  layer1.weights.assign = MatrixMaker.diagonal([5], 0);
  layer1.bias.assign = [1];
  /*layer1.weights.apply((_) => rand.nextDouble());
  layer1.bias.apply((_) => rand.nextDouble());*/
  network.addLayer(layer1);

  for (int i = 0; i < inputs.length; i++) {
    network.train(inputs[i], [y[i]]);

    print('Train$i:');
    for(int j = 0; j < network.layers.length; j++) {
      final layer = network.layers[j];
      print('Layer$j.weights:${layer.weights}, Layer$j.biases: ${layer.bias}');
      // TODO print weights print(layer.);
    }
  }

  print(network.predict(inputs[0]));
  print(y[0]);
}
