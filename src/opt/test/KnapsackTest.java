package opt.test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knap sack problem
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /**
     * Random number generator
     */
    private static final Random random = new Random();
    /**
     * The number of items
     */
    private static final int NUM_ITEMS = 40;
    /**
     * The number of copies each
     */
    private static final int COPIES_EACH = 4;
    /**
     * The maximum weight for a single element
     */
    private static final double MAX_WEIGHT = 50;
    /**
     * The maximum volume for a single element
     */
    private static final double MAX_VOLUME = 50;
    /**
     * The volume of the knapsack
     */
    private static final double KNAPSACK_VOLUME =
            MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;

    /**
     * The test main
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] weights = new double[NUM_ITEMS];
        double[] volumes = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
        EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        Integer iterations[] = {10, 50, 100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000};
        double start, end, rchr = 0, rcho = 0, sar = 0, sao = 0, gar = 0, gao = 0, mr = 0, mo = 0;

        ArrayList<String> optima = new ArrayList<String>();
        ArrayList<String> runtime = new ArrayList<String>();
        for (Integer i : iterations) {

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            start = System.nanoTime();
            new FixedIterationTrainer(rhc, i).train();
            end = System.nanoTime();
            rcho = ef.value(rhc.getOptimal());
            rchr = end - start;
            rchr /= Math.pow(10, 9);

            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            start = System.nanoTime();
            new FixedIterationTrainer(sa, i).train();
            end = System.nanoTime();
            sao = ef.value(sa.getOptimal());
            sar = end - start;
            sar /= Math.pow(10, 9);

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
            start = System.nanoTime();
            new FixedIterationTrainer(ga, i).train();
            end = System.nanoTime();
            gao = ef.value(ga.getOptimal());
            gar = end - start;
            gar /= Math.pow(10, 9);

            if (i < 100000) {
                MIMIC mimic = new MIMIC(200, 100, pop);
                start = System.nanoTime();
                new FixedIterationTrainer(mimic, i).train();
                end = System.nanoTime();
                mo = ef.value(mimic.getOptimal());
                mr = end - start;
                mr /= Math.pow(10, 9);
            }

            optima.add(i.toString() + ", " + rcho + ", " + sao + ", " + gao + ", " + mo);
            runtime.add(i.toString() + ", " + rchr + ", " + sar + ", " + gar + ", " + mr);
        }
        System.out.println("Iterations, RHC, SA, GA, MIMIC");
        for (String line : optima) System.out.println(line);
        System.out.println();
        System.out.println("Iterations, RHC, SA, GA, MIMIC");
        for (String line : runtime) System.out.println(line);
        System.out.println();

    }
}
