package demo;

import util.math.Matrix;
import neural.ToasterNeural;

public class Demo {
	public static void main(String[] args) {
		ToasterNeural n = new ToasterNeural(.3f, null, 10, 5, 10);

		Matrix input = new Matrix(new double[][]{{1, 0, 0, 0, 0, 0, 0, 0, 0, 0,}}).transpose();
		Matrix output = new Matrix(new double[][]{{1, 0, 0, 1, 0, 0, 1, 0, 0, 1}}).transpose();

		n.predict(input).getResult().print();

		for (int i = 0; i < 10000; i++) {
			System.out.println();
			n.train(input, output);
			n.predict(input).getResult().print();
		}
	}
}
