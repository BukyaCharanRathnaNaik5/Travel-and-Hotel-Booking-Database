from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("HotelBookingAnalytics").getOrCreate()

# ---- 1. Load data ----
df = spark.read.csv("data/hotel_bookings.csv", header=True, inferSchema=True)
print("Raw records:", df.count())
df.printSchema()

# ---- 2. Clean data ----
df = df.fillna({"country": "Unknown", "children": 0})
df = df.filter((F.col("adults") + F.col("children")) > 0)
df = df.filter(F.col("adr") > 0)

# ---- 3. Feature engineering ----
df = (df
    .withColumn("total_nights",
        F.col("stays_in_weekend_nights") + F.col("stays_in_week_nights"))
    .withColumn("revenue", F.col("adr") * F.col("total_nights"))
    .withColumn("lead_bucket",
        F.when(F.col("lead_time") <= 7, "0-7")
         .when(F.col("lead_time") <= 30, "8-30")
         .when(F.col("lead_time") <= 90, "31-90")
         .when(F.col("lead_time") <= 180, "91-180")
         .otherwise("180+")))
df.cache()
print("Clean records:", df.count())
df.write.mode("overwrite").parquet("outputs/clean_bookings.parquet")
df.createOrReplaceTempView("bookings")

# ---- 4. Analysis with Spark SQL ----
spark.sql("""
    SELECT hotel, COUNT(*) AS bookings,
           ROUND(AVG(is_canceled) * 100, 2) AS cancel_rate
    FROM bookings GROUP BY hotel""").show()

spark.sql("""
    SELECT arrival_date_month, hotel, COUNT(*) AS bookings,
           ROUND(AVG(CASE WHEN is_canceled = 0 THEN adr END), 2) AS avg_adr
    FROM bookings
    GROUP BY arrival_date_month, hotel""").show(24)

spark.sql("""
    SELECT market_segment, COUNT(*) AS bookings,
           ROUND(AVG(is_canceled) * 100, 2) AS cancel_rate
    FROM bookings GROUP BY market_segment
    ORDER BY bookings DESC""").show()

df.groupBy("lead_bucket").agg(
    F.round(F.avg("is_canceled") * 100, 2).alias("cancel_rate")).show()
df.groupBy("deposit_type").agg(
    F.round(F.avg("is_canceled") * 100, 2).alias("cancel_rate")).show()
