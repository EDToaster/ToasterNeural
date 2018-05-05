package demo;

import util.math.Matrix;
import neural.ToasterNeural;
import util.tools.TrainingSet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Demo {
	public static void main(String[] args) throws IOException {
		ToasterNeural n = new ToasterNeural(.3f, null, 10, 7, 3, 1);

		Random r = new Random();

		ArrayList<TrainingSet> dataset = new ArrayList<>();
		File f = new File("src/main/resources/poker_data.txt");
		BufferedReader reader = new BufferedReader(new FileReader(f));

		String s;
		while ((s = reader.readLine()) != null) {
			String[] toks = s.split("\\s*,\\s*");
			Matrix in = new Matrix(10, 1);
			Matrix out = new Matrix(1, 1);

			for (int i = 0; i < 5; i++) {
				in.set(i * 2, 0, Double.parseDouble(toks[i * 2]) / 4.0);
				in.set(i * 2 + 1, 0, Double.parseDouble(toks[i * 2 + 1]) / 13.0);
			}
			out.set(0, 0, Double.parseDouble(toks[10]) / 10.0);
			dataset.add(new TrainingSet(in, out));
		}
		reader.close();

		Scanner sc = new Scanner(System.in);

		while (true) {
			String input = sc.nextLine();
			if (input.equals("r")) {
				System.out.println("Training");
				for (int i = 0; i < 100000; i++) {
					TrainingSet train = dataset.get(r.nextInt(dataset.size()));
					n.train(train.in, train.out);
				}
				System.out.println("Training done");
				continue;
			} else {
				try {
					String[] toks = input.split("\\s*,\\s*");
					double[] converted = new double[toks.length];

					for (int i = 0; i < 5; i++) {
						converted[i * 2] = Double.parseDouble(toks[i * 2]) / 4.0;
						converted[i * 2 + 1] = Double.parseDouble(toks[i * 2 + 1]) / 13.0;
					}

					Matrix tester = new Matrix(new double[][]{converted}).transpose();
					Matrix res = n.predict(tester).getResult();
					System.out.println(res.get(0, 0) * 11);
				} catch (Exception ignored) {
				}
			}
		}
	}
}
