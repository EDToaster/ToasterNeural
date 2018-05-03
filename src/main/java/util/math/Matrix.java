package util.math;

import neural.ToasterNeural;
import util.exception.NeuralNetworkMatrixSizeException;

import java.io.Serializable;
import java.util.Random;

import static util.tools.Values.*;

public class Matrix implements Serializable {

	public static final double DEFAULT_VALUE = 0d;

	private double[][] values;
	private int m, n;

	public Matrix(double[][] values) {
		this.values = values;
		this.m = values.length;
		this.n = values[0].length;
	}

	public Matrix(int m, int n, double fill) {
		this.values = new double[m][n];
		this.m = m;
		this.n = n;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				values[i][j] = fill;
			}
		}
	}

	public Matrix(int m, int n, Random gen) {
		this(m, n);

		if (gen == null) gen = new Random();

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				values[i][j] = gen.nextFloat();
			}
		}
	}

	public Matrix(int m, int n) {
		this(m, n, DEFAULT_VALUE);
	}

	public Matrix(int m, double fill) {
		this(m, 1, fill);
	}

	public Matrix(int m) {
		this(m, 1);
	}

	// arithmetic operations
	public Matrix mult(Matrix other) {
		if (this.m() != other.n()) throw new NeuralNetworkMatrixSizeException(MATRIX_MULT);


		Matrix toReturn = new Matrix(other.m(), this.n());

		for (int i = 0; i < toReturn.n(); i++) {
			for (int j = 0; j < toReturn.m(); j++) {
				toReturn.set(j, i, dot(other, i, j));
			}
		}

		return toReturn;
	}

	public Matrix mult(double scalar) {
		return this.map(x -> x * scalar);
	}

	public Matrix hadamard(Matrix other) {
		if (this.m() != other.m() || this.n() != other.n()) throw new NeuralNetworkMatrixSizeException(MATRIX_MULT);

		Matrix toReturn = new Matrix(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				toReturn.set(i, j, get(i, j) * other.get(i, j));
			}
		}
		return toReturn;
	}

	public double dot(Matrix other) {
		if (other.isVector() && this.isVector() && this.m() == other.m()) {
			double sum = 0;
			for (int i = 0; i < this.m(); i++) {
				sum += this.values[i][0] * other.values[i][0];
			}
			return sum;
		} else {
			throw new NeuralNetworkMatrixSizeException(DOT_PRODUCT);
		}
	}

	public double dot(Matrix other, int thisColumn, int otherRow) {
		return this.spliceColumn(thisColumn).dot(other.spliceRow(otherRow));
	}

	public Matrix add(Matrix other) {
		if (this.m() != other.m() || this.n() != other.n()) throw new NeuralNetworkMatrixSizeException(MATRIX_ADD);
		Matrix toReturn = new Matrix(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				toReturn.set(i, j, get(i, j) + other.get(i, j));
			}
		}
		return toReturn;
	}

	public Matrix sub(Matrix other) {
		if (this.m() != other.m() || this.n() != other.n()) throw new NeuralNetworkMatrixSizeException(MATRIX_ADD);
		Matrix toReturn = new Matrix(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				toReturn.set(i, j, get(i, j) - other.get(i, j));
			}
		}
		return toReturn;
	}

	public Matrix spliceRow(int index) {
		return this.transpose().spliceColumn(index);
	}

	public Matrix spliceColumn(int index) {
		Matrix toReturn = new Matrix(m, 1);
		for (int i = 0; i < m; i++) {
			toReturn.set(i, 0, values[i][index]);
		}

		return toReturn;
	}

	public Matrix map(ToasterNeural.Func f) {
		Matrix toReturn = new Matrix(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				toReturn.set(i, j, f.apply(values[i][j]));
			}
		}
		return toReturn;
	}

	public double calculateDistanceCost(Matrix other) {

		if (this.m() != other.m() || this.n() != other.n()) throw new NeuralNetworkMatrixSizeException(MATRIX_COST);

		double cost = 0;

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				cost += Math.pow(values[i][j] - other.values[i][j], 2);
			}
		}
		return cost;
	}

	// getters setters

	public int m() {
		return m;
	}

	public int n() {
		return n;
	}

	public boolean isVector() {
		return n == 1;
	}

	public double get(int m, int n) {
		return values[m][n];
	}

	public void set(int m, int n, double val) {
		values[m][n] = val;
	}

	public void set(int m, double val) {
		values[m][0] = val;
	}

	public Matrix transpose() {
		Matrix toReturn = new Matrix(n, m);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				toReturn.set(j, i, values[i]
						[j]);
			}
		}
		return toReturn;
	}

	public void print() {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				System.out.printf("%.10f\t", values[i][j]);
			}
			System.out.println();
		}
	}

	public boolean equals(Object obj) {
		if (obj instanceof Matrix) {
			Matrix other = (Matrix) obj;

			if (this.m == other.m() && this.n == other.n()) {
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < m; j++) {
						if (this.get(j, i) != other.get(j, i)) {
							return false;
						}
					}
				}
				return true;
			} else {
				return false;
			}
		} else {
			return false;
		}
	}

	public static Matrix identity(int size) {
		Matrix toReturn = new Matrix(size, size, 0);
		for (int i = 0; i < size; i++) {
			toReturn.set(i, i, 1);
		}
		return toReturn;
	}

}
