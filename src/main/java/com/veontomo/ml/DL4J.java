package com.veontomo.ml;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.learning.config.Nesterovs;

/**
 * This is an example from https://deeplearning4j.org/tutorials/03-logistic-regression
 * @author Andrew
 *
 */
public class DL4J {

    public static double[] convertToBaseBits(int input, int base, int s) {
        return String.format("%" + s + "s", Integer.toString(input, base))
            .replace(' ', '0')
            .chars()
            .mapToObj(n -> Character.toString((char) n))
            .mapToInt(c -> Integer.parseInt(c))
            .mapToDouble(c -> c)
            .toArray();
    }

    public static double[] hot(int input, int s) {
        final double[] result = new double[s];
        IntStream.range(0, s)
            .forEach(i -> result[i] = i == input ? 1 : 0);
        return result;
    }

    public static void main(String[] arg) {
        final int base = 5; 
        final int N = 50000;
        final int in = Integer.toString(N, base).length();
        System.out.println(in);
        

        final List<double[]> r = IntStream.range(0, N)
            .mapToObj(i -> convertToBaseBits(i, base, in))
            .collect(Collectors.toList());
        final double[][] train = new double[r.size()][];
        IntStream.range(0, train.length)
            .forEach(i -> train[i] = r.get(i));
        final List<double[]> y = IntStream.range(0, N)
            .mapToObj(i -> hot(i % 2, 2))
            .collect(Collectors.toList());
        final double[][] label = new double[y.size()][];
        IntStream.range(0, label.length)
            .forEach(i -> label[i] = y.get(i));

        final OutputLayer outputLayer = new OutputLayer.Builder().nIn(in) // The number of inputs feed from the input layer
            .nOut(2) // The number of output values the output layer is supposed to take
            .weightInit(WeightInit.XAVIER) // The algorithm to use for weights initialization
            .activation(Activation.SOFTMAX) // Softmax activate converts the output layer into a probability distribution
            .build(); // Building

        final MultiLayerConfiguration logisticRegressionConf = new NeuralNetConfiguration.Builder().seed(1)
            .learningRate(0.001)
            .iterations(200)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(0.9)) // High Level Configuration
            .list() // For configuring MultiLayerNetwork we call the list method
            .layer(0, outputLayer) // <----- output layer fed here
            .pretrain(false)
            .backprop(true) // Pretraining and Backprop Configuration
            .build(); // Building Configuration
        DataSet allData = new DataSet(new NDArray(train), new NDArray(label));
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.85);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        final MultiLayerNetwork model = new MultiLayerNetwork(logisticRegressionConf);

        model.init();
        model.fit(trainingData);
        INDArray output = model.output(testData.getFeatureMatrix());
        Evaluation eval = new Evaluation(2);
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());
    }
}
