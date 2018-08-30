// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.FastTree;
using Microsoft.ML.Runtime.FastTree.Internal;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

[assembly: LoadableClass(typeof(FastTreeParallelBinCheckpointChecker),
    null, typeof(Microsoft.ML.Runtime.FastTree.SignatureParallelTrainer), "checkpointer")]

namespace Microsoft.ML.Runtime.RunTests
{
    using SplitInfo = Microsoft.ML.Runtime.FastTree.Internal.LeastSquaresRegressionTreeLearner.SplitInfo;
    using LeafSplitCandidates = Microsoft.ML.Runtime.FastTree.Internal.LeastSquaresRegressionTreeLearner.LeafSplitCandidates;

    public static class SharedParameters
    {
        public static double[][] BinUpperBounds;
        public static Ensemble TreeEnsemble;
        public static int NumIterations;

        public static void Persist()
        {
            if (BinUpperBounds != null)
                PersistBins();
            if (TreeEnsemble != null)
                PersistEnsemble();
        }

        private static void PersistBins()
        {
            double[][] binUpperBounds = new Double[BinUpperBounds.Length][];
            for (int i = 0; i < BinUpperBounds.Length; i++)
            {
                binUpperBounds[i] = new double[BinUpperBounds[i].Length];
                BinUpperBounds[i].CopyTo(binUpperBounds[i], 0);
            }
            // Point to the data copy
            BinUpperBounds = binUpperBounds;
        }

        private static void PersistEnsemble()
        {
            var ensemble = new Ensemble();
            foreach (var tree in TreeEnsemble.Trees)
            {
                // Just copy the tree pointer for now
                ensemble.AddTree(tree);
            }
            TreeEnsemble = ensemble;
        }
    }

    internal sealed class FastTreeParallelBinCheckpointChecker : FastTreeParallelCheckpointCheckerBase
    {
        public FastTreeParallelBinCheckpointChecker(ITestOutputHelper logger) : base(logger)
        {
        }

        public override bool InitializeBins(double[][] binUpperBounds)
        {
            if (SharedParameters.BinUpperBounds != null)
            {
                Logger.WriteLine("TEST #3: Copying bins to FastTree");
                // Copy the bins if they exist
                for (int i = 0; i < SharedParameters.BinUpperBounds.Length; i++) {
                    binUpperBounds[i] = new double[SharedParameters.BinUpperBounds[i].Length];
                    SharedParameters.BinUpperBounds[i].CopyTo(binUpperBounds[i], 0);
                }
                return true;
            }

            Logger.WriteLine("TEST #1: Not copying bins; saving a reference for later!");
            // Otherwise, keep a pointer
            SharedParameters.BinUpperBounds = binUpperBounds;
            return false;
        }
    }

    internal sealed class FastTreeParallelEnsembleCheckpointChecker : FastTreeParallelCheckpointCheckerBase
    {
        public FastTreeParallelEnsembleCheckpointChecker(ITestOutputHelper logger) : base(logger)
        {
        }

        public override void InitializeTraining(Ensemble ensemble)
        {
            if (SharedParameters.TreeEnsemble != null)
            {
                var howManyTreesToCopy = SharedParameters.TreeEnsemble.NumTrees - 2;
                Logger.WriteLine("TEST #3: Copying {0} of {1} trees from Ensemble to FastTree", howManyTreesToCopy, SharedParameters.TreeEnsemble.NumTrees);
                // Copy the bins if they exist
                for (int i = 0; i < howManyTreesToCopy; i++)
                {
                    ensemble.AddTree(SharedParameters.TreeEnsemble.GetTreeAt(i));
                }
                return;
            }

            Logger.WriteLine("TEST #1: Not copying ensemble; saving a reference for later!");
            // Otherwise, keep a pointer
            SharedParameters.TreeEnsemble = ensemble;
        }
    }

    [TlcModule.Component(Name = "ParallelTrainingFactoryForChecking")]
    public sealed class ParallelTrainingFactoryForChecking : ISupportParallelTraining
    {
        private readonly ITestOutputHelper Logger;

        public enum TestType { Bin, Ensemble };
        private static TestType _testType;

        public ParallelTrainingFactoryForChecking(ITestOutputHelper logger, TestType testType)
        {
            Logger = logger;
            _testType = testType;
        }

        public IParallelTraining CreateComponent(IHostEnvironment env)
        {
            if (_testType == TestType.Bin)
                return new FastTreeParallelBinCheckpointChecker(Logger);
            else if (_testType == TestType.Ensemble)
                return new FastTreeParallelEnsembleCheckpointChecker(Logger);
            else
                throw new NotImplementedException();
        }
    }

    public class TestParallelFasttreeCheckpoints : BaseTestBaseline
    {
        private readonly ITestOutputHelper Logger;

        public TestParallelFasttreeCheckpoints(ITestOutputHelper logger)
            : base(logger)
        {
            Logger = logger;
        }

        [Fact]
        [TestCategory("ParallelFasttree")]
        public void CheckParallelFastTreeBinCheckpoint()
        {
            ISupportParallelTraining parallelTrainingFactory = 
                new ParallelTrainingFactoryForChecking(Logger, ParallelTrainingFactoryForChecking.TestType.Bin);

            using (var env = new TlcEnvironment())
            {
                var firstTrainer = new FastTreeBinaryClassificationTrainer(env, new FastTreeBinaryClassificationTrainer.Arguments()
                {
                    NumTrees = 5,
                    ParallelTrainer = parallelTrainingFactory
                });
                var firstPredictor = TrainPredictor(env, firstTrainer, out RoleMappedData dataset);

                var secondTrainer = new FastTreeBinaryClassificationTrainer(env, new FastTreeBinaryClassificationTrainer.Arguments()
                {
                    NumTrees = 5,
                    ParallelTrainer = parallelTrainingFactory
                });
                var secondPredictor = TrainPredictor(env, secondTrainer, out dataset);

                // Compare the predictors
                ComparePredictors(env, firstPredictor, secondPredictor, dataset);
            }
        }

        [Fact]
        [TestCategory("ParallelFasttree")]
        public void CheckParallelFastTreeEnsembleCheckpoint()
        {
            ISupportParallelTraining parallelTrainingFactory = 
                new ParallelTrainingFactoryForChecking(Logger, ParallelTrainingFactoryForChecking.TestType.Ensemble);

            using (var env = new TlcEnvironment())
            {
                var firstTrainer = new FastTreeBinaryClassificationTrainer(env, new FastTreeBinaryClassificationTrainer.Arguments()
                {
                    NumTrees = 5,
                    ParallelTrainer = parallelTrainingFactory
                });
                var firstPredictor = TrainPredictor(env, firstTrainer, out RoleMappedData dataset);

                var secondTrainer = new FastTreeBinaryClassificationTrainer(env, new FastTreeBinaryClassificationTrainer.Arguments()
                {
                    NumTrees = 5,
                    ParallelTrainer = parallelTrainingFactory
                });
                var secondPredictor = TrainPredictor(env, secondTrainer, out dataset);

                // Compare the predictors
                ComparePredictors(env, firstPredictor, secondPredictor, dataset);
            }
        }

        private IPredictor TrainPredictor(TlcEnvironment env, FastTreeBinaryClassificationTrainer trainer, out RoleMappedData dataset)
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var outRoot = @"..\Common\CheckPointTest";

            var modelOutPath = DeleteOutputPath(outRoot, "codegen-model.zip");
            var csOutPath = DeleteOutputPath(outRoot, "codegen-out.cs");

            // Pipeline

            var loader = new TextLoader(env,
                new TextLoader.Arguments()
                {
                    Column = new[]
                    {
                        new TextLoader.Column()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoader.Range() { Min=0, Max=0} },
                            Type = DataKind.R4
                        },
                        new TextLoader.Column()
                        {
                            Name = "Features",
                            Source = new [] { new TextLoader.Range() { Min=1, Max=9} },
                            Type = DataKind.R4
                        }
                    }
                },
                new MultiFileSource(dataPath));

            // Specify the dataset
            dataset = new RoleMappedData(loader, label: "Label", feature: "Features");

            // Train on the dataset
            return trainer.Train(dataset);
        }

        private void ComparePredictors(TlcEnvironment env, IPredictor firstPredictor, IPredictor secondPredictor, RoleMappedData dataset)
        {
            var firstModel = GetModel(env, firstPredictor, dataset);
            var firstPredictions = firstModel.Predict(GetTestData(), false).Select(p => p.Probability);

            var secondModel = GetModel(env, secondPredictor, dataset);
            var secondPredictions = secondModel.Predict(GetTestData(), false).Select(p => p.Probability);

            Assert.Equal(firstPredictions, secondPredictions);
            Logger.WriteLine("Prediction comparison passed!");
        }

        private BatchPredictionEngine<TestData, Prediction> GetModel(TlcEnvironment env, IPredictor predictor, RoleMappedData dataset)
        {
            var scorer = ScoreUtils.GetScorer(predictor, dataset, env, dataset.Schema);
            return env.CreateBatchPredictionEngine<TestData, Prediction>(scorer);
        }

        public class TestData
        {
            [Column(ordinal: "0", name: "Label")]
            public float Label;

            [Column(ordinal: "1-9")]
            [VectorType(9)]
            public float[] Features;
        }

        public class Prediction
        {
            [ColumnName("Probability")]
            public float Probability;
        }

        private IEnumerable<TestData> GetTestData()
        {
            return new[]
            {
                new TestData
                {
                    Features = new float[9] {0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,}
                },
                new TestData
                {
                    Features = new float[9] {0f, 1f, 0f, 0f, -100f, 1000f, 0f, 0f, 0f,}
                },
                new TestData
                {
                    Features = new float[9] {0f, 1f, 1f, 0f, 10f, 0f, 0f, -123f, 0f,}
                },
                new TestData
                {
                    Features = new float[9] {0f, 1f, 1f, 1f, 0f, 78f, 0f, 0f, 0f,}
                },
                new TestData
                {
                    Features = new float[9] {0f, 1f, 0.345f, 1f, 1f, 0f, 0f, 0f, 0f,}
                },
                new TestData
                {
                    Features = new float[9] {1f, 1f, 1f, 1f, 1f, 1f, 1e-4f, 1f, 2f,}
                }
            };
        }
    }

    internal class FastTreeParallelCheckpointCheckerBase : IParallelTraining
    {
        protected readonly ITestOutputHelper Logger;

        public FastTreeParallelCheckpointCheckerBase(ITestOutputHelper logger)
        {
            Logger = logger;
        }

        public virtual bool InitializeBins(double[][] binUpperBounds)
        {
            return false;
        }

        public virtual void InitializeTraining(Ensemble ensemble)
        {
            return;
        }

        public void CacheHistogram(bool isSmallerLeaf, int featureIdx, int subfeature, SufficientStatsBase sufficientStatsBase, bool HasWeights)
        {
            return;
        }

        public bool IsNeedFindLocalBestSplit()
        {
            return true;
        }

        public void FindGlobalBestSplit(LeafSplitCandidates smallerChildSplitCandidates,
            LeafSplitCandidates largerChildSplitCandidates,
            Microsoft.ML.Runtime.FastTree.FindBestThresholdFromRawArrayFun findFunction,
            SplitInfo[] bestSplits)
        {
            return;
        }

        public void GetGlobalDataCountInLeaf(int leafIdx, ref int cnt)
        {
            return;
        }

        public bool[] GetLocalBinConstructionFeatures(int numFeatures)
        {
            return Utils.CreateArray<bool>(numFeatures, true);
        }

        public double[] GlobalMean(Dataset dataset, RegressionTree tree, DocumentPartitioning partitioning, double[] weights, bool filterZeroLambdas)
        {
            double[] means = new double[tree.NumLeaves];
            for (int l = 0; l < tree.NumLeaves; ++l)
            {
                means[l] = partitioning.Mean(weights, dataset.SampleWeights, l, filterZeroLambdas);
            }
            return means;
        }

        public void PerformGlobalSplit(int leaf, int lteChild, int gtChild, SplitInfo splitInfo)
        {
            return;
        }

        public void InitIteration(ref bool[] activeFeatures)
        {
            SharedParameters.NumIterations++;
            return;
        }

        public void InitEnvironment()
        {
            SharedParameters.NumIterations = 0;
            return;
        }

        public void InitTreeLearner(Dataset trainData, int maxNumLeaves, int maxCatSplitPoints, ref int minDocInLeaf)
        {
            return;
        }

        public RegressionTree LearnTree(IChannel ch, bool[] activeFeatures, Func<IChannel, bool[], double[], RegressionTree> fitTargets, 
            Func<IChannel, double[]> computeTargets, Action<int> setSeed)
        {
            setSeed(123);
            return fitTargets(ch, activeFeatures, computeTargets(ch));
        }

        public void SyncGlobalBoundary(int numFeatures, int maxBin, Double[][] binUpperBounds)
        {
            return;
        }

        public virtual void FinalizeEnvironment()
        {
            Logger.WriteLine("Test INFO: {0} trees were fit.", SharedParameters.NumIterations);
            Logger.WriteLine("TEST #2: Persisting state for the next round!");
            SharedParameters.Persist();
            return;
        }

        public void FinalizeTreeLearner()
        {
            return;
        }

        public void FinalizeIteration()
        {
            return;
        }

        public bool IsSkipNonSplittableHistogram()
        {
            return true;
        }
    }
}
