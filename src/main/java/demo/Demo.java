package demo;

import util.math.Matrix;
import neural.ToasterNeural;
import util.tools.TrainingSet;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Demo {
	public static void main(String[] args) {
		ToasterNeural n = new ToasterNeural(.3f, null, 1, 3, 1);

		Random r = new Random();
		Scanner sc = new Scanner(System.in);

		while (true) {
			String input = sc.nextLine();
			if (input.equals("re")) {
				for (int i = 0; i < 100000; i++) {
					double s = Math.random();
					n.train(new Matrix(new double[][]{{s}}).transpose(), new Matrix(new double[][]{{Math.abs(Math.sin(s))}}).transpose());
				}
				continue;
			}

			String[] inputs = input.split("\\s*,\\s*");
			double[] doubleArr = new double[inputs.length];

			for (int i = 0; i < inputs.length; i++) {
				doubleArr[i] = Double.parseDouble(inputs[i]);
			}

			n.predict(new Matrix(new double[][]{doubleArr}).transpose()).getResult().print();
		}
	}
}
