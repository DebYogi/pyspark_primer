"""Feature engineering module"""
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

class FeatureEngineer:
    def __init__(self, feature_cols, label_col):
        self.feature_cols = feature_cols
        self.label_col = label_col

    def get_stages(self):
        label_indexer = StringIndexer(inputCol=self.label_col, outputCol="label")
        assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features_assembled")
        scaler = StandardScaler(inputCol="features_assembled", outputCol="features", withMean=True, withStd=True)
        return [label_indexer, assembler, scaler]
