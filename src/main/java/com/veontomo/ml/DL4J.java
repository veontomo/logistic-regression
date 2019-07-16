package com.veontomo.ml;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;

/**
 * This is an example from https://deeplearning4j.org/tutorials/03-logistic-regression
 * @author Andrew
 *
 */
public class DL4J {

    public void main(String[] arg) {
        final OutputLayer outputLayer = new OutputLayer.Builder()
            .nIn(784) //The number of inputs feed from the input layer
            .nOut(10) //The number of output values the output layer is supposed to take
            .weightInit(WeightInit.XAVIER) //The algorithm to use for weights initialization
            .activation(Activation.SOFTMAX) //Softmax activate converts the output layer into a probability distribution
            .build(); //Building
        
        final MultiLayerConfiguration logisticRegressionConf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .learningRate(0.1).iterations(1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Nesterovs(0.9)) //High Level Configuration
            .list() //For configuring MultiLayerNetwork we call the list method
            .layer(0, outputLayer) //    <----- output layer fed here
            .pretrain(false).backprop(true) //Pretraining and Backprop Configuration
            .build(); //Building Configuration
    }
}
