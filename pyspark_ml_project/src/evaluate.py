"""Evaluation module"""
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class Evaluator:
    def __init__(self, label_col="label", prediction_col="prediction", metric="accuracy"):
        self.evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName=metric)

    def evaluate(self, df):
        return self.evaluator.evaluate(df)
