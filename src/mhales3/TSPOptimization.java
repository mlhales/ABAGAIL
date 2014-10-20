package mhales3;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.Instance;

public class TSPOptimization {
    /** The n value */
    private static final int N = 14;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        TravelingSalesmanEvaluationFunction mef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(mef, odd, df);

        Integer iterations[] = {100, 500, 1000, 5000, 10000, 50000, 100000};
        double start, end, rchr = 0, rcho = 0, sar = 0, sao = 0, gar = 0, gao = 0, mr = 0, mo = 0;

        ArrayList<String> optima = new ArrayList<String>();
        ArrayList<String> runtime = new ArrayList<String>();
        for (Integer i : iterations) {

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            start = System.nanoTime();
            new FixedIterationTrainer(rhc, i).train();
            end = System.nanoTime();
            rcho = -ef.value(rhc.getOptimal());
            rchr = end - start;
            rchr /= Math.pow(10, 9);

            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
            start = System.nanoTime();
            new FixedIterationTrainer(sa, i).train();
            end = System.nanoTime();
            sao = -ef.value(sa.getOptimal());
            sar = end - start;
            sar /= Math.pow(10, 9);

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(400, 200, 20, gap);
            start = System.nanoTime();
            new FixedIterationTrainer(ga, i).train();
            end = System.nanoTime();
            gao = -ef.value(ga.getOptimal());
            gar = end - start;
            gar /= Math.pow(10, 9);

            MIMIC mimic = new MIMIC(200, 20, pop);
            start = System.nanoTime();
            new FixedIterationTrainer(mimic, i).train();
            end = System.nanoTime();
            mo = -ef.value(mimic.getOptimal());
            mr = end - start;
            mr /= Math.pow(10, 9);

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
