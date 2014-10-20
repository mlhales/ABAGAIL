package mhales3;

import opt.OptimizationAlgorithm;

public class OptimizerTrainer {

    /**
     * The trainer
     */
    private OptimizationAlgorithm optimizer;

    /**
     * The threshold
     */
    private double threshold;

    /**
     * The number of iterations trained
     */
    private int iterations;

    /**
     * The maximum number of iterations to use
     */
    private int maxIterations;

    /**
     * Create a new optimizer trainer
     * @param optimizer the optimizer to use
     * @param threshold the error threshold
     * @param maxIterations the maximum iterations
     */
    public OptimizerTrainer(OptimizationAlgorithm optimizer,
                            double threshold, int maxIterations) {
        this.optimizer = optimizer;
        this.threshold = threshold;
        this.maxIterations = maxIterations;
    }

    public double train() {
        double optimum = Double.MAX_VALUE;
        do {
            iterations++;
            optimizer.train();
            optimum = optimizer.getOptimum();
        } while (optimum < threshold
                && iterations < maxIterations);
        return optimum;
    }

    /**
     * Get the number of iterations used
     * @return the number of iterations
     */
    public int getIterations() {
        return iterations;
    }


}
