import 'dart:math';

import 'package:ann/ann.dart';

void main() {
  final rand = Random(5);

  final dense = Dense(10, 5);
  dense.weights.apply((_) => rand.nextDouble());
  dense.bias.apply((_) => rand.nextDouble());

  print(dense.compute(List<double>.generate(10, (index) => rand.nextDouble())));
}
