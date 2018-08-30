// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    //An interface that can be implemnted on
    public interface IFastTrainingScoresUpdate
    {
        ScoreTracker GetUpdatedTrainingScores();
    }

    public abstract class OptimizationAlgorithm
    {
        //TODO: We should move Partitioning to OptimizationAlgorithm
        public TreeLearner TreeLearner;

        public ObjectiveFunctionBase ObjectiveFunction;

        // This is added to signalize that we are just about to update all scores
        // This is only used fof printing training graph scores that we can compute fast for the previous iteration saving topLables by scores from n+1 gradient computation
        public delegate void PreScoreUpdateHandler(IChannel ch);
        public PreScoreUpdateHandler PreScoreUpdateEvent;

        public Ensemble Ensemble;

        public ScoreTracker TrainingScores;
        public List<ScoreTracker> TrackedScores;

        public IStepSearch AdjustTreeOutputsOverride; // if set it overrides IStepSearch possibly implemented by ObejctiveFunctionBase
        public double Smoothing;
        protected double DropoutRate;
        protected Random DropoutRng;
        private int _dropoutSeed;
        public bool UseFastTrainingScoresUpdate;

        protected readonly IParallelTraining ParallelTraining;

        public OptimizationAlgorithm(Ensemble ensemble, Dataset trainData, double[] initTrainScores,
            IParallelTraining parallelTraining, double dropoutRate = 0, int dropoutSeed = int.MinValue)
        {
            Ensemble = ensemble;
            TrainingScores = ConstructScoreTracker("train", trainData, initTrainScores);
            TrackedScores = new List<ScoreTracker>();
            TrackedScores.Add(TrainingScores);
            if (dropoutSeed != int.MinValue)
                DropoutRng = new Random(dropoutSeed);
            _dropoutSeed = dropoutSeed;
            DropoutRate = dropoutRate;
            UseFastTrainingScoresUpdate = true;
            ParallelTraining = parallelTraining;
        }

        public void SetTrainingData(Dataset trainData, double[] initTrainScores)
        {
            TrainingScores = ConstructScoreTracker("train", trainData, initTrainScores);
            TrackedScores[0] = TrainingScores;
        }

        public abstract RegressionTree TrainingIteration(IChannel ch, bool[] activeFeatures);
        //Regularize a regression tree with smoothing paramter alpha

        public virtual void UpdateAllScores(IChannel ch, RegressionTree tree)
        {
            PreScoreUpdateEvent?.Invoke(ch);
            using (Timer.Time(TimerEvent.UpdateScores))
            {
                foreach (ScoreTracker t in TrackedScores)
                    UpdateScores(t, tree);
            }
        }

        /// <summary>
        /// Update all the scores using the ensemble. Used when there is an update
        /// to the ensemble outside of the usual control-flow.
        /// </summary>
        public virtual void UpdateAllScoresInEnsemble(IChannel ch, int offset=0)
        {
            for (int i = offset; i < Ensemble.NumTrees; ++i)
            {
                PreScoreUpdateEvent?.Invoke(ch);
                foreach (ScoreTracker t in TrackedScores)
                    // Directly add all scores to all sets
                    t.AddScores(Ensemble.GetTreeAt(i), 1.0);
            }
        }

        public virtual void UpdateScores(ScoreTracker t, RegressionTree tree)
        {
            if (t == TrainingScores)
            {
                IFastTrainingScoresUpdate fastUpdate = AdjustTreeOutputsOverride as IFastTrainingScoresUpdate;
                ScoreTracker updatedScores = (UseFastTrainingScoresUpdate && fastUpdate != null) ? fastUpdate.GetUpdatedTrainingScores() : null;
                if (updatedScores != null)
                    t.SetScores(updatedScores.Scores);
                else
                    t.AddScores(tree, TreeLearner.Partitioning, 1.0);
            }
            else
                t.AddScores(tree, 1.0);
        }

        public ScoreTracker GetScoreTracker(string name, Dataset set, double[] initScores)
        {
            //Fisrt check for duplicates maybe we already track scores for set dataset
            foreach (var st in TrackedScores) if (st.Dataset == set)
                    return st;

            ScoreTracker newTracker = ConstructScoreTracker(name, set, initScores);
            //add the constructed tracker to the list of scores we need to update
            TrackedScores.Add(newTracker);
            return newTracker;
        }

        protected abstract ScoreTracker ConstructScoreTracker(string name, Dataset set, double[] initScores);

        protected virtual void SmoothTree(RegressionTree tree, double smoothing)
        {
            if (smoothing == 0.0)
                return;

            //Create recursive structure of the tree starting from root node
            var regularizer = new RecursiveRegressionTree(tree, TreeLearner.Partitioning, 0);

            //Perform bottom-up computation of weighted interior node output
            double rootNodeOutput = regularizer.GetWeightedOutput();
            //followed by top-down propagation of parent's output value
            regularizer.SmoothLeafOutputs(rootNodeOutput, smoothing);
        }

        public virtual void FinalizeLearning(int bestIteration)
        {
            if (bestIteration != Ensemble.NumTrees)
            {
                Ensemble.RemoveAfter(Math.Max(bestIteration, 0));
                TrackedScores.Clear();  //Invalidate all precomputed scores as they are not valid anymore //slow method of score computation will be used instead
            }
        }

        protected void ResetDropoutSeed()
        {
            DropoutRng = new Random(_dropoutSeed + Iteration);
        }

        protected int Iteration => Ensemble.NumTrees;
    }
}
