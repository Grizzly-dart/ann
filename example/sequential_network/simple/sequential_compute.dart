import 'dart:math';

import 'package:grizzly_ann/grizzly_ann.dart';

void main() {
  final rand = Random(5);
  final x1 = rand.ints(100, max: 10).toList();
  final x2 = rand.ints(100, max: 10).toList();

  final y = (((x1 * 5) + 17) + ((x2 * 8) + 19)).toList();

  final data = MatrixMaker.fromColumns([x1, x2, y]);

  final inputs = MatrixMaker.fromColumns([x1, x2]);

  final network = Sequential1DNetwork();

  final layer1 = Dense(2, 2);
  layer1.weights.assign = MatrixMaker.diagonal([5, 8], 0);
  layer1.bias.assign = [17, 19];
  network.addLayer(layer1);

  final layer2 = Dense(2, 1);
  layer2.weights.assign = MatrixMaker.fromColumns([[1, 1]]);
  layer2.bias.assign = [0];
  network.addLayer(layer2);

  print('${inputs[0]}, ${y[0]}');
  print(network.predict(inputs[0]));
}
