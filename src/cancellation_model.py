from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)

spark = SparkSession.builder.appName("CancellationModel").getOrCreate()
df = spark.read.parquet("outputs/clean_bookings.parquet")

cat_cols = ["hotel", "market_segment", "deposit_type", "arrival_date_month"]
num_cols = ["lead_time", "is_repeated_guest", "previous_cancellations",
            "total_of_special_requests", "required_car_parking_spaces",
            "adr", "total_nights", "adults", "children"]

indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx",
                          handleInvalid="keep") for c in cat_cols]
encoder = OneHotEncoder(inputCols=[c + "_idx" for c in cat_cols],
                        outputCols=[c + "_vec" for c in cat_cols])
assembler = VectorAssembler(
    inputCols=num_cols + [c + "_vec" for c in cat_cols],
    outputCol="features")

train, test = df.randomSplit([0.8, 0.2], seed=42)

models = {
    "Logistic Regression": LogisticRegression(
        featuresCol="features", labelCol="is_canceled", maxIter=100),
    "Random Forest": RandomForestClassifier(
        featuresCol="features", labelCol="is_canceled",
        numTrees=100, maxDepth=12, seed=42),
}

auc_eval = BinaryClassificationEvaluator(
    labelCol="is_canceled", metricName="areaUnderROC")

for name, algo in models.items():
    pipe = Pipeline(stages=indexers + [encoder, assembler, algo])
    pred = pipe.fit(train).transform(test)
    print(name, "AUC:", round(auc_eval.evaluate(pred), 3))
    for metric in ["accuracy", "precisionByLabel",
                   "recallByLabel", "fMeasureByLabel"]:
        ev = MulticlassClassificationEvaluator(
            labelCol="is_canceled", metricName=metric, metricLabel=1.0)
        print(" ", metric, round(ev.evaluate(pred), 3))
