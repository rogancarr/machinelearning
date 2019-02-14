// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.RunTests;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Transforms.TensorFlow;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    public class Evaluation : BaseTestClass
    {

        public Evaluation(ITestOutputHelper output) : base(output)
        {
        }
        
        /// <summary>
        /// Reconfigurable predictions: The following should be possible: A user trains a binary classifier,
        /// and through the test evaluator gets a PR curve, the based on the PR curve picks a new threshold
        /// and configures the scorer (or more precisely instantiates a new scorer over the same model parameters)
        /// with some threshold derived from that.
        /// </summary>
        [Fact]
        public void ReconfigurablePrediction()
        {
            var mlContext = new MLContext(seed: 789);

            // Get the dataset, create a train and test
            var data = mlContext.Data.CreateTextLoader(TestDatasets.housing.GetLoaderColumns(), hasHeader: true)
                .Read(BaseTestClass.GetDataPath(TestDatasets.housing.trainFilename));
            var split = mlContext.BinaryClassification.TrainTestSplit(data, testFraction: 0.2);

            // Create a pipeline to train on the housing data
            var pipeline = mlContext.Transforms.Concatenate("Features", new string[] {
                    "CrimesPerCapita", "PercentResidental", "PercentNonRetail", "CharlesRiver", "NitricOxides", "RoomsPerDwelling",
                    "PercentPre40s", "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio"})
                .Append(mlContext.Transforms.CopyColumns("Label", "MedianHomeValue"))
                .Append(mlContext.Regression.Trainers.OrdinaryLeastSquares());

            var model = pipeline.Fit(split.TrainSet);

            var scoredTest = model.Transform(split.TestSet);
            var metrics = mlContext.Regression.Evaluate(scoredTest);

            Common.CheckMetrics(metrics);

            // Todo #2465: Allow the setting of threshold and thresholdColumn for scoring.
            // This is no longer possible in the API
            //var newModel = new BinaryPredictionTransformer<IPredictorProducing<float>>(ml, model.Model, trainData.Schema, model.FeatureColumn, threshold: 0.01f, thresholdColumn: DefaultColumnNames.Probability);
            //var newScoredTest = newModel.Transform(pipeline.Transform(testData));
            //var newMetrics = mlContext.BinaryClassification.Evaluate(scoredTest);
            // And the Threshold and ThresholdColumn properties are not settable.
            //var predictor = model.LastTransformer;
            //predictor.Threshold = 0.01; // Not possible
        }

        /// <summary>
        /// An existing TF model can be used to produce predictions.
        /// </summary>
        [TensorFlowFact]
        public void TensorFlowTransforCifarEndToEndTest()
        {
            var imageHeight = 32;
            var imageWidth = 32;
            var model_location = "cifar_model/frozen_model.pb";
            var dataFile = GetDataPath("images/images.tsv");
            var imageFolder = Path.GetDirectoryName(dataFile);
            var outputLength = 10;

            var mlContext = new MLContext(seed: 1, conc: 1);
            var data = mlContext.Data.CreateTextLoader(new[] {
                        new TextLoader.Column("ImagePath", DataKind.TX, 0),
                        new TextLoader.Column("Label", DataKind.TX, 1),
                    }).Read(dataFile);

            var schema = TensorFlowUtils.GetModelSchema(mlContext, model_location);

            // Create a pipeline to read in images and score them with a TensorFlow model
            var pipeline = mlContext.Transforms.LoadImages(imageFolder, ("ImageReal", "ImagePath"))
                    .Append(mlContext.Transforms.Resize(new ImageResizingEstimator.ColumnInfo("ImageCropped", imageHeight, imageWidth, "ImageReal")))
                    .Append(mlContext.Transforms.ExtractPixels(new ImagePixelExtractingEstimator.ColumnInfo("Input", "ImageCropped", interleave: true)))
                    .Append(mlContext.Transforms.ScoreTensorFlowModel(model_location, "Output", "Input"));
                    //.Append(mlContext.Transforms.Concatenate("Features", "Output"))
                    //.Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                    //.AppendCacheCheckpoint(mlContext)
                    //.Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent();
            var tensorFlowModel = pipeline.Fit(data);

            // Get the predictions of the model
            var predictions = tensorFlowModel.Transform(data);
            var predictionEnumerable = mlContext.CreateEnumerable<CifarPrediction>(predictions, true);

            // Validate the outputs of the model
            var last = new float[outputLength];
            foreach (var prediction in predictionEnumerable)
            {
                // Validate that the prediction is of the correct length
                Assert.Equal(outputLength, prediction.Output.Length);

                // Validate that it's not just outputing Zeros
                Assert.NotEqual(0, Sum(prediction.Output));

                Common.AssertNotEqual(last, prediction.Output);
                prediction.Output.CopyTo(last, 0);
            }





            //var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
            //Assert.Equal(1, metrics.AccuracyMicro, 2);

            //var predictFunction = transformer.CreatePredictionEngine<ImageClassificationData, TensorFlowPrediction>(mlContext);
            //var prediction = predictFunction.Predict(new ImageClassificationData()
            //{
            //    ImagePath = GetDataPath("images/banana.jpg")
            //});
            //Assert.Equal(0, prediction.PredictedScores[0], 2);
            //Assert.Equal(1, prediction.PredictedScores[1], 2);
            //Assert.Equal(0, prediction.PredictedScores[2], 2);

            //prediction = predictFunction.Predict(new ImageClassificationData()
            //{
            //    ImagePath = GetDataPath("images/hotdog.jpg")
            //});
            //Assert.Equal(0, prediction.PredictedScores[0], 2);
            //Assert.Equal(0, prediction.PredictedScores[1], 2);
            //Assert.Equal(1, prediction.PredictedScores[2], 2);
        }

        private double Sum(float[] array)
        {
            double sum = 0.0;
            for (int i = 0; i < array.Length; i++)
                sum += array[i];
            return sum;
        }
    }
}
