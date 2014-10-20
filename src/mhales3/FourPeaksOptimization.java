package mhales3;

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

public class FourPeaksOptimization {

    public static void main(String[] args) {

        Integer n_values[] = {30, 50, 70, 100};
        double start, end, rchr = 0, rcho = 0, sar = 0, sao = 0, gar = 0, gao = 0, mr = 0, mo = 0;

        ArrayList<String> optima = new ArrayList<String>();
        ArrayList<String> runtime = new ArrayList<String>();
        for (Integer n: n_values) {

            int[] ranges = new int[n];
            int t = n / 10;
            double optimum = (2.0 * n) - t - 1.0;
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new FourPeaksEvaluationFunction(t);
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            start = System.nanoTime();
            OptimizerTrainer rhct = new OptimizerTrainer(rhc, optimum, 200000);
            rhct.train();
            end = System.nanoTime();
            rcho = rhct.getIterations();
            rchr = end - start;
            rchr /= Math.pow(10,9);

            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
            start = System.nanoTime();
            OptimizerTrainer sat = new OptimizerTrainer(sa, optimum, 200000);
            sat.train();
            end = System.nanoTime();
            sao = sat.getIterations();
            sar = end - start;
            sar /= Math.pow(10,9);

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
            start = System.nanoTime();
            OptimizerTrainer gat = new OptimizerTrainer(ga, optimum, 200000);
            gat.train();
            end = System.nanoTime();
            gao = gat.getIterations();
            gar = end - start;
            gar /= Math.pow(10,9);

            MIMIC mimic = new MIMIC(200, 20, pop);
            start = System.nanoTime();
            OptimizerTrainer mt = new OptimizerTrainer(mimic, optimum, 20000);
            mt.train();
            end = System.nanoTime();
            mo = mt.getIterations();
            mr = end - start;
            mr /= Math.pow(10,9);

            optima.add(n.toString() + ", " + rcho + ", " + sao + ", " + gao + ", " + mo);
            runtime.add(n.toString() + ", " + rchr + ", " + sar  + ", " + gar  + ", " + mr);
        }
        System.out.println("N, RHC, SA, GA, MIMIC");
        for (String line :optima) System.out.println(line);
        System.out.println();
        System.out.println("N, RHC, SA, GA, MIMIC");
        for (String line :runtime) System.out.println(line);
        System.out.println();

    }
}
