import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

//stfu
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("logs.csv")

// Predict the label, based on the rest of the data.
// For this example, we want to predict the server load based on time, memutil and netutil.
//"hostname", "hour", "load", "memutil", "netutil"]
val df = data.select(data("load").as("label"),$"hour",$"memutil",$"netutil")

// VectorAssembler
val assembler = new VectorAssembler().setInputCols(Array("hour","memutil","netutil")).setOutputCol("features")
val output = assembler.transform(df).select($"label",$"features")

val lr = new LinearRegression()
val lrModel = lr.fit(output)

val trainSum= lrModel.summary
trainSum.residuals.show()
println(s"Root Mean Squared Error ${trainingSummary.rootMeanSquaredError}")
println(s"Summary R2 ${trainingSummary.r2}")
