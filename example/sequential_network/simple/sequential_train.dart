import 'dart:math';

import 'package:grizzly_ann/grizzly_ann.dart';

void main() {
  final rand = Random(5);
  final x1 = rand.ints(100, max: 10).toList();
  final x2 = rand.ints(100, max: 10).toList();

  final y = (((x1 * 5) + 17) + ((x2 * 8) + 19)).toDoubles().toList();

  final data = MatrixMaker.fromColumns([x1, x2, y]);

  final inputs = MatrixMaker.fromColumns([x1, x2]);

  final network = Sequential1DNetwork();

  final layer1 = Dense(2, 2);
  layer1.weights.assign = MatrixMaker.diagonal([5, 8], 0);
  layer1.bias.assign = [17, 19];
  /*layer1.weights.apply((_) => rand.nextDouble());
  layer1.bias.apply((_) => rand.nextDouble());*/
  network.addLayer(layer1);

  final layer2 = Dense(2, 1);
  layer2.weights.assign = MatrixMaker.fromColumns([[2, 1]]);
  layer2.bias.assign = [0];
  /*layer2.weights.apply((_) => rand.nextDouble());
  layer2.bias.apply((_) => rand.nextDouble());*/
  network.addLayer(layer2);

  for (int i = 0; i < inputs.length; i++) {
    network.train(inputs[i], [y[i]]);
  }

  for(final layer in network.layers) {
    print(layer.weights);
    // TODO print weights print(layer.);
  }

  print(network.predict(inputs[0]));
  print(y[0]);
}
