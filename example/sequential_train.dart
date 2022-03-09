import 'dart:math';

import 'package:ann/ann.dart';
import 'package:grizzly_array/grizzly_array.dart';

void main() {
  final rand = Random(5);
  final x1 = Double1D.gen(100, (index) => rand.nextInt(100).toDouble());
  final x2 = Double1D.gen(100, (index) => rand.nextInt(100).toDouble());
  
  final y = ((x1 * 5) + 17) * 2 + ((x2 * 8) + 19) * 3;
  
  final data = Double2D.fromColumns([x1, x2, y]);

  final inputs = Double2D.fromColumns([x1, x2]);

  final network = Sequential1DNetwork();

  final layer1 = Dense(2, 2);
  layer1.weights.apply((_) => rand.nextDouble());
  layer1.bias.apply((_) => rand.nextDouble());
  network.addLayer(layer1);

  final layer2 = Dense(2, 1);
  layer2.weights.apply((_) => rand.nextDouble());
  layer2.bias.apply((_) => rand.nextDouble());
  network.addLayer(layer2);

  for(int i = 0; i < x1.length; i++) {
    network.train(inputs[i], Double1D([y[i]]));
  }

  for(final layer in network.layers) {
    print(layer.);
  }

  print(network.predict(inputs[0]));
  print(y[0]);
}
