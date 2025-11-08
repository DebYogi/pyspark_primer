"""
main.py
Main entry point for the PySpark ML OOP project with Logistic Regression, GBT, Keras, and XGBoost

Run this from inside src/:
    cd my_pyspark_course/src
    source ../venv/bin/activate
    python main.py
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline as SparkPipeline

# Local imports (files inside src/)
from src import config, data, preprocessing, evaluate
from src import model_new as model

# -----------------------------
# 1) Start Spark session
# -----------------------------
spark = SparkSession.builder \
    .appName("pyspark_ml_oop_project") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

print("✅ Spark session started")
print("Spark version:", spark.version)

# -----------------------------
# 2) Load data
# -----------------------------
loader = data.DataLoader(spark, config.DATA_URL, config.DATA_LOCAL_PATH)
loader.download_data()
df = loader.load_csv()

# -----------------------------
# 3) Feature engineering (create stages)
# -----------------------------
fe = preprocessing.FeatureEngineer(config.FEATURE_COLUMNS, config.LABEL_COLUMN)
stages = fe.get_stages()

# -----------------------------
# 4) Train/test split
# -----------------------------
train_df, test_df = df.randomSplit([config.TRAIN_RATIO, config.TEST_RATIO], seed=config.SEED)
print(f"Training samples: {train_df.count()}, Test samples: {test_df.count()}")

# -----------------------------
# 4.1) Apply preprocessing-only pipeline to produce 'features' and 'label'
# -----------------------------
preproc_pipeline = SparkPipeline(stages=stages)
preproc_model = preproc_pipeline.fit(train_df)    # fit only on training set
train_prepared = preproc_model.transform(train_df)
test_prepared = preproc_model.transform(test_df)

print("Prepared train schema (showing transformed columns):")
train_prepared.printSchema()
train_prepared.select("features", "label").show(3, truncate=False)

# -----------------------------
# 5) Logistic Regression (Spark ML pipeline)
# -----------------------------
trainer_lr = model.ModelTrainer(model_type="logreg", model_params=config.LR_PARAMS)
# build a full pipeline (we could reuse stages, but preprocessing already applied above;
# here we build pipeline including preprocessors so we can demonstrate both ways)
# Option A: Fit via pipeline on raw train_df:
trainer_lr.build_pipeline(stages)
trainer_lr.fit(train_df)
preds_lr = trainer_lr.predict(test_df)
evaluator = evaluate.Evaluator()
acc_lr = evaluator.evaluate(preds_lr)
print(f"Logistic Regression Accuracy: {acc_lr:.4f}")

# -----------------------------
# 6) Gradient Boosted Trees (Spark ML)
# -----------------------------
trainer_gbt = model.ModelTrainer(model_type="gbt", model_params=config.GBT_PARAMS)
trainer_gbt.build_pipeline(stages)
trainer_gbt.fit(train_df)
preds_gbt = trainer_gbt.predict(test_df)
acc_gbt = evaluator.evaluate(preds_gbt)
print(f"GBT (OneVsRest) Accuracy: {acc_gbt:.4f}")

# -----------------------------
# 7) Keras Neural Network (trained on pandas extracted features)
# -----------------------------
trainer_keras = model.ModelTrainer(model_type="keras", model_params={"epochs": 50, "batch_size": 8, "verbose": 0})
trainer_keras.fit(train_prepared)                  # note: use preprocessed Spark DF with 'features' & 'label'
preds_keras = trainer_keras.predict(test_prepared) # returns Spark DataFrame with 'prediction'
acc_keras = evaluator.evaluate(preds_keras)
print(f"Keras Neural Network Accuracy: {acc_keras:.4f}")

# -----------------------------
# 8) XGBoost (trained on pandas extracted features)
# -----------------------------
trainer_xgb = model.ModelTrainer(model_type="xgboost", model_params={"num_round": 50})
trainer_xgb.fit(train_prepared)
preds_xgb = trainer_xgb.predict(test_prepared)
acc_xgb = evaluator.evaluate(preds_xgb)
print(f"XGBoost Accuracy: {acc_xgb:.4f}")

# -----------------------------
# 9) Stop Spark
# -----------------------------
spark.stop()
print("✅ Spark session stopped. Done.")
