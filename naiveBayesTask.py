from pyspark.sql import SparkSession
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint


spark = SparkSession.Builder().getOrCreate()

filename = r"C:\Users\Miltos\Desktop\winequality-red.csv"

data = spark.read.csv(filename, inferSchema=True, header=True)

rdd_data = data.rdd.map(lambda line: LabeledPoint(line[11], line[0:10]))

training, test = rdd_data.randomSplit([0.6, 0.4], seed=0)

model = NaiveBayes.train(training, 1.0)

predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = predictionAndLabel.filter(lambda x: x[0] == x[1]).count() / test.count()

print('model accuracy {}'.format(accuracy))
