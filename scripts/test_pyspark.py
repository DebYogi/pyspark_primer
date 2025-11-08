from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("pyspark-mac-test") \
    .master("local[*]") \
    .getOrCreate()

print("Spark Version:", spark.version)

data = [(1, "Alice"), (2, "Bob")]
df = spark.createDataFrame(data, ["id", "name"])
df.show()

spark.stop()
