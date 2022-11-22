package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author Sarah Shader and Jia Wu
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;
	
	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;
	
	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight
	
	protected int iterations = 10;
	
	protected int loss = EXPONENTIAL_LOSS;
	protected int regularization = NO_REGULARIZATION;
	protected double eta = 0.01;
	protected double lambda = 0.01;

	/**
	 * Get a weight vector over the set of features with each weight
	 * set to 0
	 * 
	 * @param features the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features){
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();
		
		for( Integer f: features){
			temp.put(f, 0.0);
		}
		
		return temp;
	}
	
	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features){
		weights = getZeroWeights(features);
		b = 0;
	}
	
	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations){
		this.iterations = iterations;
	}


	public void setLoss(int loss) {
		switch (loss) {
			case 0: this.loss = EXPONENTIAL_LOSS;
					break;
			case 1: this.loss = HINGE_LOSS;
					break; 
			default: break;
		}
	}

	public void setRegularization(int regularization) {
		switch (regularization) {
			case 0: this.regularization = NO_REGULARIZATION;
					break;
			case 1: this.regularization = L1_REGULARIZATION;
					break;
			case 2: this.regularization = L2_REGULARIZATION;
					break; 
			default: break;
		}
	}

	public void setLambda(double lambda){
		this.lambda = Math.abs(lambda); 
	}

	public void setEta(double eta){
		this.eta = Math.abs(eta); 
	}
	
	/**
	 * calculates loss function 
	 * @param e example 
	 * @return the loss function results 
	 */
	protected double getC(Example e) {
		double label = e.getLabel();
		double yPrime = getDistanceFromHyperplane(e, weights, b);
		
		if (loss == EXPONENTIAL_LOSS) {
			return Math.exp(-1 * label * yPrime);
		} else {
			if (label * yPrime < 1) {
				return 1;
			} else {
				return 0;
			}
		}
	}

	/**
	 * calculates regulator based on weight of index j 
	 * @param w_j
	 * @return regulator for weight of index j 
	 */
	protected double getR(double w_j) {
		if (regularization == NO_REGULARIZATION) {
			return 0.0;
		} else if (regularization == L1_REGULARIZATION) {
			return Math.signum(w_j);
		} else { // L2_REGULARIZATION
			return w_j;
		}
	}
	
	/**
	 * minimize the objective function by a number of iterations by picking a random point
	 * and then stepping towards the minimum
	 * @param data
	 */
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());
		
		ArrayList<Example> training = (ArrayList<Example>) data.getData().clone();
		
		for( int it = 0; it < iterations; it++ ){
			Collections.shuffle(training);
			for( Example e: training ){
				double c = getC(e);
				double y = e.getLabel();

				// update the weights
				for( Integer featureIndex: e.getFeatureSet() ){
					double r = getR(weights.get(featureIndex));
					double weight_j = weights.get(featureIndex);
					double featureValue = e.getFeature(featureIndex);
					
					weight_j += eta * (y * featureValue * c - lambda * r);
					weights.put(featureIndex, weight_j);
				}
				
				// update b
				b += eta * (y * c - lambda * getR(b));	
			}
		}
	}

	private double calculateWeightMagnitude(HashMap<Integer, Double> weights){
		double sum = 0; 

		for(double w: weights.values()){
			sum += Math.pow(w, 2); 
		}

		return sum; 
	}

	private double getHingeLoss(Example e) {
		// l(y, y') = max(0,1− yy')
		
		double label = e.getLabel();
		double yPrime = getDistanceFromHyperplane(e, weights, b);
		double loss = Math.max(0, 1 - label * yPrime); 

		// update the weights
		if(regularization == L2_REGULARIZATION){
			loss += (lambda/2) * calculateWeightMagnitude(weights); 
		}
		
		return loss; 
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example);
	}
	
	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

		
	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e the example to predict
	 * @return
	 */
	protected double getPrediction(Example e){
		return getPrediction(e, weights, b);
	}
	
	/**
	 * Get the prediction from the on this example from using weights w and inputB
	 * 
	 * @param e example to predict
	 * @param w the set of weights to use
	 * @param inputB the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB){
		double sum = getDistanceFromHyperplane(e,w,inputB);

		if( sum > 0 ){
			return 1.0;
		}else if( sum < 0 ){
			return -1.0;
		}else{
			return 0;
		}
	}
	
	protected static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB){
		double sum = inputB;
		
		//for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for( Integer featureIndex: e.getFeatureSet()){
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}
		
		return sum;
	}
	
	public String toString(){
		StringBuffer buffer = new StringBuffer();
		
		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);
		
		for(Integer index: temp){
			buffer.append(index + ":" + weights.get(index) + " ");
		}
		
		return buffer.substring(0, buffer.length()-1);
	}

	public GradientDescentClassifier(){

	}
}
