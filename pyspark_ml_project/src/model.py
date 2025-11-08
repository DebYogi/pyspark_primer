"""Model training module"""
from pyspark.ml.classification import LogisticRegression, GBTClassifier, OneVsRest
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class ModelTrainer:
    def __init__(self, model_type="logreg", model_params=None):
        self.model_type = model_type
        self.model_params = model_params if model_params else {}
        self.pipeline = None
        self.model = None

    def build_pipeline(self, stages):
        if self.model_type == "logreg":
            estimator = LogisticRegression(featuresCol="features", labelCol="label", **self.model_params)
        elif self.model_type == "gbt":
            gbt = GBTClassifier(featuresCol="features", labelCol="label", **self.model_params)
            estimator = OneVsRest(classifier=gbt, labelCol="label", featuresCol="features")
        else:
            raise ValueError("Unsupported model type")

        self.pipeline = Pipeline(stages=stages + [estimator])
        return self.pipeline

    def fit(self, train_df, param_grid=None, folds=3):
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        if param_grid:
            crossval = CrossValidator(estimator=self.pipeline, estimatorParamMaps=param_grid,
                                      evaluator=evaluator, numFolds=folds, parallelism=2)
            self.model = crossval.fit(train_df)
        else:
            self.model = self.pipeline.fit(train_df)
        return self.model

    def predict(self, df):
        return self.model.transform(df)
