package demo;

import neural.ToasterNeural;
import org.math.plot.Plot2DPanel;
import org.math.plot.plots.Plot;
import org.math.plot.utils.Array;
import util.math.Matrix;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.Arrays;

import static java.lang.Math.PI;

public class FunctionDemo {

    private static final ToasterNeural.Func arctan = x -> Math.atan(16 * x) / PI;
    private static final ToasterNeural.Func sin = x -> Math.sin(16 * x) / 2 + 0.5;
    private static final ToasterNeural.Func cos = x -> Math.cos(16 * x) / 2 + 0.5;
    private static final ToasterNeural.Func bin = x -> (int) (10 * x) % 2 == 0 ? 0 : 1;
    private static final ToasterNeural.Func floor = x -> Math.floor(4 * x) / 3;
    private static final ToasterNeural.Func heart = x -> Math.sin(x) * Math.sin(10 * x) * 0.7 + 0.5;
    private static final ToasterNeural.Func random = x -> Math.random() * x;
    private static final ToasterNeural.Func binx = x -> (int) (6 * x) % 2 == 0 ? 1 - x : x;


    public static void main(String[] args) {
        ToasterNeural n = new ToasterNeural(.1f, null, 1, 8, 1);

        double range = 1;
        int testing = 1000;

        ToasterNeural.Func funcToUse = cos;


        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = initPlot();

        int divs = 100;
        double[] calculated = new double[divs];
        double[] inx = new double[divs];

        for (int i = 0; i < divs; i++) {
            float x = (float) (i / (float) divs * range);
            inx[i] = x;
            calculated[i] = Math.abs(funcToUse.apply(x));
        }

        plot.addLinePlot("Calculated", inx, calculated);
        int p = -1;

        int epochs = 0;

        while (true) {
            System.out.println("Epoch " + epochs++);

            for (int i = 0; i < testing; i++) {
                float x = (float) Math.random() * (float) range;

                n.train(
                        new Matrix(new double[][]{{x}}),
                        new Matrix(new double[][]{{Math.abs(funcToUse.apply(x))}}));
            }

            double[] guessed = new double[divs];

            for (int i = 0; i < divs; i++) {
                float x = (float) (i / (float) divs * range);
                guessed[i] = n.predict(new Matrix(new double[][]{{x}})).getResult().get(0, 0);
            }

            if (p == -1) {
                p = plot.addLinePlot("NN", inx, guessed);
            } else {
                plot.changePlotData(p, Array.mergeColumns(inx, guessed));
            }
            plot.repaint();

            // add a line plot to the PlotPanel
            plot.addLegend(Plot2DPanel.EAST);
        }
    }

    private static Plot2DPanel initPlot() {
        Plot2DPanel plot = new Plot2DPanel();

        // put the PlotPanel in a JFrame, as a JPanel
        JFrame frame = new JFrame("function learning");
        frame.setContentPane(plot);
        frame.setSize(new Dimension(800, 600));
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        return plot;
    }

}
