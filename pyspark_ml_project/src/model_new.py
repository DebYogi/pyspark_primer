"""
model.py
Model training module extended with Keras and XGBoost support.

Supported model_type values:
 - "logreg"  : PySpark LogisticRegression (via pipeline)
 - "gbt"     : PySpark GBTClassifier wrapped with OneVsRest
 - "keras"   : Keras neural network (trained on Pandas DataFrame)
 - "xgboost" : XGBoost (trained on Pandas DataFrame)

Notes:
 - For "keras" and "xgboost" we expect the input Spark DataFrame to already have
   'features' (Vector) and 'label' (numeric) columns. Convert using a preprocessing
   pipeline before calling fit().
 - Predictions for external models are returned as Spark DataFrames containing the
   original columns plus a numeric 'prediction' column.
"""

from pyspark.ml.classification import LogisticRegression, GBTClassifier, OneVsRest
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# External libs
import pandas as pd
import numpy as np

# Keras
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.utils import to_categorical
except Exception:
    Sequential = None  # We'll raise helpful errors if user tries to train Keras without TF installed

# XGBoost
try:
    import xgboost as xgb
except Exception:
    xgb = None


class ModelTrainer:
    def __init__(self, model_type="logreg", model_params=None):
        """
        model_type: "logreg", "gbt", "keras", "xgboost"
        model_params: dict of hyperparameters for chosen model
        """
        self.model_type = model_type
        self.model_params = model_params if model_params else {}
        self.pipeline = None        # Spark ML Pipeline (when applicable)
        self.model = None           # Fitted model (PipelineModel or external model)

    # --------------------------
    # Build Spark ML pipeline for logreg/gbt
    # --------------------------
    def build_pipeline(self, stages):
        """
        stages: list of Spark ML transformers (e.g., label indexer, assembler, scaler)
        Returns: Pipeline or None (for keras/xgboost)
        """
        if self.model_type == "logreg":
            estimator = LogisticRegression(featuresCol="features", labelCol="label", **self.model_params)
        elif self.model_type == "gbt":
            gbt = GBTClassifier(featuresCol="features", labelCol="label", **self.model_params)
            estimator = OneVsRest(classifier=gbt, labelCol="label", featuresCol="features")
        elif self.model_type in ["keras", "xgboost"]:
            # External trainers do not use Spark ML pipelines (we assume preprocessing already applied)
            return None
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.pipeline = Pipeline(stages=stages + [estimator])
        return self.pipeline

    # --------------------------
    # Fit
    # --------------------------
    def fit(self, train_df, param_grid=None, folds=3):
        """
        train_df: Spark DataFrame. For keras/xgboost, must have 'features' and 'label'.
        param_grid: optional parameter grid (only used for Spark ML models)
        folds: number of CV folds for CrossValidator
        """
        if self.model_type in ["logreg", "gbt"]:
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            if param_grid:
                crossval = CrossValidator(estimator=self.pipeline, estimatorParamMaps=param_grid,
                                          evaluator=evaluator, numFolds=folds, parallelism=self.model_params.get("parallelism", 2))
                self.model = crossval.fit(train_df)
            else:
                self.model = self.pipeline.fit(train_df)
            return self.model

        elif self.model_type == "keras":
            if Sequential is None:
                raise ImportError("TensorFlow/Keras is not available in the environment. Install tensorflow to use 'keras' model_type.")

            # Convert Spark DataFrame to pandas for training
            # Expecting 'features' column is a Vector; convert to list-of-lists
            pdf = train_df.select("features", "label").toPandas()
            X = pd.DataFrame(pdf["features"].tolist())
            y = pdf["label"].astype(int).values
            input_dim = X.shape[1]
            output_dim = len(np.unique(y))

            # Build model
            model = Sequential([
                Dense(self.model_params.get("hidden1", 32), activation='relu', input_dim=input_dim),
                Dense(self.model_params.get("hidden2", 16), activation='relu'),
                Dense(output_dim, activation='softmax')
            ])
            model.compile(optimizer=self.model_params.get("optimizer", "adam"),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            y_cat = to_categorical(y)
            model.fit(X, y_cat,
                      epochs=self.model_params.get("epochs", 50),
                      batch_size=self.model_params.get("batch_size", 8),
                      verbose=self.model_params.get("verbose", 1))
            self.model = model
            return self.model

        elif self.model_type == "xgboost":
            if xgb is None:
                raise ImportError("xgboost is not available in the environment. Install xgboost to use 'xgboost' model_type.")

            pdf = train_df.select("features", "label").toPandas()
            X = pd.DataFrame(pdf["features"].tolist())
            y = pdf["label"].astype(int).values
            dtrain = xgb.DMatrix(X, label=y)

            params = self.model_params.get("params", {"objective": "multi:softprob", "num_class": int(np.max(y) + 1)})
            num_round = self.model_params.get("num_round", 50)
            self.model = xgb.train(params, dtrain, num_round)
            return self.model

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    # --------------------------
    # Helper: convert predicted labels array to Spark DataFrame with 'prediction' column
    # --------------------------
    @staticmethod
    def _preds_array_to_spark(df_original, predicted_labels):
        """
        df_original: Spark DataFrame (prepared) — we will attach monotonic row id, join predictions,
                     and return Spark DataFrame with prediction column.
        predicted_labels: 1D array-like (same order as df_original.toPandas())
        """
        from pyspark.sql import functions as F
        spark = df_original.sparkSession

        # Add a stable numeric id to the Spark DF (monotonically_increasing_id)
        df_with_id = df_original.withColumn("__row_id", F.monotonically_increasing_id())

        # Extract the row ids in pandas order so alignment is preserved
        rowid_pd = df_with_id.select("__row_id").toPandas()
        preds_pd = pd.DataFrame({"__row_id": rowid_pd["__row_id"].values, "prediction": predicted_labels})

        # Create spark DF of predictions and join back
        preds_spark = spark.createDataFrame(preds_pd)
        joined = df_with_id.join(preds_spark, on="__row_id").drop("__row_id")

        # Ensure prediction column is numeric type double (evaluator expects numeric)
        joined = joined.withColumn("prediction", joined["prediction"].cast("double"))
        return joined

    # --------------------------
    # Predict
    # --------------------------
    def predict(self, df):
        """
        df: Spark DataFrame.
            - For logreg/gbt: pipeline model.transform(df) -> returns Spark DataFrame (with 'prediction').
            - For keras/xgboost: df must have 'features' column; returns Spark DataFrame with 'prediction' column.
        """
        if self.model_type in ["logreg", "gbt"]:
            # PipelineModel.transform returns Spark DataFrame with prediction column already
            return self.model.transform(df)

        elif self.model_type == "keras":
            if self.model is None:
                raise ValueError("Model not fitted. Call fit() before predict().")

            pdf = df.select("features").toPandas()
            X = pd.DataFrame(pdf["features"].tolist())
            probs = self.model.predict(X)
            predicted_labels = np.argmax(probs, axis=1)
            return self._preds_array_to_spark(df, predicted_labels)

        elif self.model_type == "xgboost":
            if self.model is None:
                raise ValueError("Model not fitted. Call fit() before predict().")

            pdf = df.select("features").toPandas()
            X = pd.DataFrame(pdf["features"].tolist())
            dtest = xgb.DMatrix(X)
            probs = self.model.predict(dtest)
            # xgboost predict returns shape (n_rows, n_classes)
            predicted_labels = np.argmax(probs, axis=1)
            return self._preds_array_to_spark(df, predicted_labels)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")