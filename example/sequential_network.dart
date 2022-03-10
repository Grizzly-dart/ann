import 'dart:math';

import 'package:grizzly_ann/grizzly_ann.dart';

void main() {
  final rand = Random(5);

  final network = Sequential1DNetwork();

  final layer1 = Dense(10, 5);
  layer1.weights.apply((_) => rand.nextDouble());
  layer1.bias.apply((_) => rand.nextDouble());
  network.addLayer(layer1);

  final layer2 = Dense(5, 2);
  layer2.weights.apply((_) => rand.nextDouble());
  layer2.bias.apply((_) => rand.nextDouble());
  network.addLayer(layer2);

  print(
      network.predict(List<double>.generate(10, (index) => rand.nextDouble())));
}
