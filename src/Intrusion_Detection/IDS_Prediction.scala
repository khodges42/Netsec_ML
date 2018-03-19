import org.apache.spark.ml._
import org.apache.log4j._
import spark.implicits._
Logger.getLogger("org").setLevel(Level.ERROR)



val spark = SparkSession.builder().getOrCreate()
//val data = spark.read.option("multiline","true").json("input.json")
var now = System.nanoTime
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("data_full.csv")
val model = PipelineModel.read.load("../../models/IDS")
val results = model.transform(data)
var timeElapsed = System.nanoTime - now

val count = results.count
val seconds = timeElapsed / 1000000000.0;
val secsperlog = seconds / count
println(f"$count predictions in $seconds seconds")
println(f"$secsperlog seconds per row" )
