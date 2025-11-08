"""Main entry point for the PySpark ML OOP project"""
from pyspark.sql import SparkSession
from pathlib import Path
from src import config, data, preprocessing, model, evaluate

spark = SparkSession.builder.appName("pyspark_ml_oop_project").master("local[*]").getOrCreate()

loader = data.DataLoader(spark, config.DATA_URL, config.DATA_LOCAL_PATH)
loader.download_data()
df = loader.load_csv()

fe = preprocessing.FeatureEngineer(config.FEATURE_COLUMNS, config.LABEL_COLUMN)
stages = fe.get_stages()

train_df, test_df = df.randomSplit([config.TRAIN_RATIO, config.TEST_RATIO], seed=config.SEED)

trainer = model.ModelTrainer(model_type="logreg", model_params=config.LR_PARAMS)
pipeline = trainer.build_pipeline(stages)
model_fitted = trainer.fit(train_df)

preds = trainer.predict(test_df)
evaluator = evaluate.Evaluator()
acc = evaluator.evaluate(preds)
print(f"Logistic Regression Accuracy: {acc:.4f}")

trainer_gbt = model.ModelTrainer(model_type="gbt", model_params=config.GBT_PARAMS)
pipeline_gbt = trainer_gbt.build_pipeline(stages)
model_gbt = trainer_gbt.fit(train_df)
preds_gbt = trainer_gbt.predict(test_df)
acc_gbt = evaluator.evaluate(preds_gbt)
print(f"GBT (OneVsRest) Accuracy: {acc_gbt:.4f}")

spark.stop()
