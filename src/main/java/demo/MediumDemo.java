package demo;

import neural.ToasterNeural;
import util.math.Matrix;
import util.tools.PredictionSet;

import static java.lang.Math.PI;

/**
 * This demo uses a neural network to emulate the sin function
 */
public class MediumDemo {

    // needs abs() because output needs to be bound to [0, 1]
    private static ToasterNeural.Func sin = x -> Math.abs(Math.sin(x));

    public static void main(String[] args) {
        ToasterNeural tn = new ToasterNeural(1, 8, 1); // create neural net with 1 in and 1 out

        int epochs = 200;           // How many rounds
        int trainingsPerEpoch = 10; // How many data points to train with per round
        int testingsPerEpoch = 100; // How many data points to test with per round

        for (int i = 0; i < epochs; i++) {

            double errorSum = 0; // running sum of error values

            // Training the neural network
            for (int j = 0; j < trainingsPerEpoch; j++) {

                // generate rand ∈ [0, PI)
                double rand = Math.random() * PI;

                // figure out correct answer
                double calculatedResult = sin.apply(rand);

                // Creating input vector
                Matrix in = new Matrix(
                        new double[][]{{rand}}
                );

                // Creating output vector
                Matrix out = new Matrix(
                        new double[][]{{calculatedResult}}
                );


                // Train using random number and correct answer
                tn.train(in, out);
            }

            // Testing the neural network
            for (int j = 0; j < testingsPerEpoch; j++) {
                double rand = Math.random() * PI;
                double calculatedResult = sin.apply(rand);

                // Guess using random data point
                PredictionSet guess = tn.predict(rand);

                // Add errors to running sum
                errorSum += guess.getResult().get(0, 0) - calculatedResult;
            }

            // Find average errors in this round and output
            System.out.println("Epoch: " + i + "\tError: " + errorSum / testingsPerEpoch);
        }
    }
}
