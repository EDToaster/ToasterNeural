package util.tools;

import util.exception.NeuralNetworkMatrixSizeException;
import util.math.Matrix;

import static util.tools.Values.CONFORM_VECTOR;

public class TrainingSet {
	public Matrix in;
	public Matrix out;

	public TrainingSet(Matrix in, Matrix out) {
		if (in.n() != 1 || out.n() != 1) throw new NeuralNetworkMatrixSizeException(CONFORM_VECTOR);
		this.in = in;
		this.out = out;
	}
}
