import 'dart:math';

import 'package:grizzly_ann/grizzly_ann.dart';

void main() {
  final rand = Random(5);
  final x1 = rand.ints(10, max: 10).toList();
  print(x1);

  final y = (x1 * 5 + 2).toList();

  final inputs = MatrixMaker.fromColumn(x1);
  print(inputs);
  print(y);

  final network = Sequential1DNetwork();

  final layer1 = Dense(1, 1);
  layer1.weights.assign = MatrixMaker.diagonal([5], 0);
  layer1.bias.assign = [2];
  network.addLayer(layer1);

  print('${inputs[0]}, ${y[0]}');
  print(network.predict(inputs[0]));
}
