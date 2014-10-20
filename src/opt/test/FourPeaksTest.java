package opt.test;

import java.util.ArrayList;
import java.util.Arrays;

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
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 100;
    /** The t value */
    private static final int T = N / 10;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        Integer iterations[] = {10, 100, 1000, 2000, 4000, 8000, 10000, 12000, 14000, 16000, 18000, 20000};
        double start, end, rchr = 0, rcho = 0, sar = 0, sao = 0, gar = 0, gao = 0, mr = 0, mo = 0;

        ArrayList<String> optima = new ArrayList<String>();
        ArrayList<String> runtime = new ArrayList<String>();
        for (Integer i: iterations) {

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            start = System.nanoTime();
            new FixedIterationTrainer(rhc, i).train();
            end = System.nanoTime();
            rcho = ef.value(rhc.getOptimal());
            rchr = end - start;
            rchr /= Math.pow(10,9);

            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
            start = System.nanoTime();
            new FixedIterationTrainer(sa, i).train();
            end = System.nanoTime();
            sao = ef.value(sa.getOptimal());
            sar = end - start;
            sar /= Math.pow(10,9);

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(400, 200, 20, gap);
            start = System.nanoTime();
            new FixedIterationTrainer(ga, i).train();
            end = System.nanoTime();
            gao = ef.value(ga.getOptimal());
            gar = end - start;
            gar /= Math.pow(10,9);
            if(i < 4000) {
                MIMIC mimic = new MIMIC(200, 20, pop);
                start = System.nanoTime();
                new FixedIterationTrainer(mimic, i).train();
                end = System.nanoTime();
                mo = ef.value(mimic.getOptimal());
                mr = end - start;
                mr /= Math.pow(10,9);
            }
            optima.add(i.toString() + ", " + rcho + ", " + sao + ", " + gao + ", " + mo);
            runtime.add(i.toString() + ", " + rchr + ", " + sar  + ", " + gar  + ", " + mr);
        }
        System.out.println("Iterations, RHC, SA, GA, MIMIC");
        for (String line :optima) System.out.println(line);
        System.out.println();
        System.out.println("Iterations, RHC, SA, GA, MIMIC");
        for (String line :runtime) System.out.println(line);
        System.out.println();
    }
}
