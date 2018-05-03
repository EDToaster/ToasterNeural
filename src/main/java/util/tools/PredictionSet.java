package util.tools;

import util.math.Matrix;

import java.util.ArrayList;

public class PredictionSet {
	public ArrayList<Matrix> layers;

	{
		layers = new ArrayList<>();
	}

	public void add(Matrix m) {
		layers.add(m);
	}

	public Matrix getResult() {
		return layers.get(layers.size() - 1);
	}

	public Matrix getInput() {
		return layers.get(0);
	}

	/**
	 * Computes the reverse of the layers, starting from output -> input
	 *
	 * @return an array of layers
	 */
	public Matrix[] getInvertedLayers() {
		Matrix[] reversed = new Matrix[layers.size()];
		for (int i = layers.size() - 1; i >= 0; i--) {
			reversed[reversed.length - i - 1] = layers.get(i);
		}
		return reversed;
	}
}
