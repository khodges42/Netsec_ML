import org.apache.spark.ml._
import org.apache.log4j._
import spark.implicits._
Logger.getLogger("org").setLevel(Level.ERROR)



val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("multiline","true").json("input.json")

val model = PipelineModel.read.load("../../models/IDS")

val results = model.transform(data)
