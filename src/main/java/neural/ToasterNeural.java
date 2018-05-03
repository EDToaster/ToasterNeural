package neural;

import util.exception.NeuralNetworkLayerNumberException;
import util.exception.NeuralNetworkMatrixSizeException;
import util.math.Matrix;
import util.tools.PredictionSet;

import java.io.Serializable;
import java.util.Random;

import static util.tools.Values.*;

public class ToasterNeural implements Serializable {

	private Matrix[] cWeights; //connector weights
	private Matrix[] nBiases; // node biases
	private double learningRate;


	public ToasterNeural(double learningRate, Random r, int... nodes) {

		if (nodes.length < 2) {
			throw new NeuralNetworkLayerNumberException(TOO_FEW_LAYERS_EXCEPTION_MESSAGE);
		}

		if (r == null) r = new Random();
		this.learningRate = learningRate;

		cWeights = new Matrix[nodes.length - 1];
		nBiases = new Matrix[nodes.length - 1];

		for (int i = 0; i < nodes.length - 1; i++) {
			int in = nodes[i];
			int out = nodes[i + 1];

			Matrix connectors = new Matrix(out, in, r);
			Matrix bias = new Matrix(out, 1, r);
			cWeights[i] = connectors;
			nBiases[i] = bias;
		}
	}

	public ToasterNeural(int... nodes) {
		this(DEFAULT_LEARNING_RATE, null, nodes);
	}

	public PredictionSet predict(Matrix input) {
		if (input.m() != cWeights[0].n())
			throw new NeuralNetworkMatrixSizeException(INPUT_VECTOR_LENGTH_EXCEPTION_MESSAGE);

		Matrix curr = input;
		PredictionSet set = new PredictionSet();
		set.add(curr);

		for (int i = 0; i < cWeights.length; i++) {
			curr = curr.mult(cWeights[i]).add(nBiases[i]).map(sigmoid.f);
			set.add(curr);
			assert curr.isVector();
		}

		return set;
	}

	public void train(Matrix input, Matrix target) {

		if (target.m() != cWeights[cWeights.length - 1].m())
			throw new NeuralNetworkMatrixSizeException(TARGET_VECTOR_LENGTH_EXCEPTION_MESSAGE);


		PredictionSet result = predict(input);
		Matrix[] reversedLayers = result.getInvertedLayers();

		Matrix lastErrors = null;

		for (int i = 0; i < reversedLayers.length - 1; i++) {

			Matrix prevLayerTransposed = reversedLayers[i + 1].transpose();
			Matrix layer = reversedLayers[i];

			Matrix transitionLayer = cWeights[cWeights.length - i - 1];
			// feed forward matrix transforms previous layer to this layer.
			Matrix errors = null;

			if (i == 0) {
				errors = target.sub(layer);
			} else {
				Matrix prev = cWeights[cWeights.length - i];
				errors = lastErrors.mult(prev.transpose());
			}

			lastErrors = errors;

			Matrix gradients = layer.map(sigmoid.df).hadamard(errors).mult(learningRate);

			Matrix transitionDeltas = prevLayerTransposed.mult(gradients);

			cWeights[cWeights.length - i - 1] = transitionLayer.add(transitionDeltas);
			nBiases[nBiases.length - i - 1] = nBiases[nBiases.length - i - 1].add(gradients);

		}
	}

	public void printWeights() {
		for (Matrix cWeight : cWeights) {
			System.out.printf("-----%d×%d%s", cWeight.m(), cWeight.n(), System.lineSeparator());
			cWeight.print();
		}
		System.out.println("-----");
	}


	public static final ActivatorFunction sigmoid = new ActivatorFunction(
			(x -> 1.0 / (1.0 + Math.exp(-x))),    // f
			(y -> y * (1 - y))                    // df
	);


	public interface Func {
		public double apply(double in);
	}

	public static class ActivatorFunction {
		public Func f, df;

		public ActivatorFunction(Func f, Func df) {
			this.f = f;
			this.df = df;
		}
	}
}
