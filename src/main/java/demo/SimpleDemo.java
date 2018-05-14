package demo;

import neural.ToasterNeural;

import java.util.Scanner;

/**
 * This demo tries to predict 1 - x (poorly)
 */
public class SimpleDemo {

    private static ToasterNeural.Func oneMinusX = x -> 1 - x;

    public static void main(String[] args) {
        ToasterNeural tn = new ToasterNeural(1, 2, 1); // one input, one output, 2 hidden

        // Training
        for (int i = 0; i < 10000; i++) {
            double rand = Math.random();
            tn.train(rand, oneMinusX.apply(rand));
        }

        // Takes user input and outputs a prediction
        Scanner sc = new Scanner(System.in);
        while (true) {
            double input = sc.nextDouble();
            System.out.println(tn.predict(input).getResult().get(0, 0)); // prints out result
        }

        /*

        >> .8
        0.1867944895721184
        >> .2
        0.8134136190181894

         */
    }
}
