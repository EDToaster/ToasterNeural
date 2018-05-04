package util.tools;

public class Values {
	public static final String TOO_FEW_LAYERS_EXCEPTION_MESSAGE = "The amount of nodes in the neural network must exceed 2.";
	public static final String INPUT_VECTOR_LENGTH_EXCEPTION_MESSAGE = "The neural network input data size is inconsistent with matrix size.";
	public static final String TARGET_VECTOR_LENGTH_EXCEPTION_MESSAGE = "The neural network target data size is inconsistent with matrix size.";
	public static final String MATRIX_MULT = "Matrix size is inconsistent when multiplying.";
	public static final String MATRIX_ADD = "Matrix size is inconsistent when adding.";
	public static final String MATRIX_COST = "Matrix size is inconsistent when calculating cost function.";
	public static final String DOT_PRODUCT = "Dot products must be between two m×1 vectors.";
	public static final String CONFORM_VECTOR = "Must input m×1 vectors.";

	public static final double DEFAULT_LEARNING_RATE = 0.1d;
}
