
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StandardScaler,VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics


// stfu
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("data_canadAI.csv")


var logselect = data.select(data(" Label").as("label_unreg"),
                            $" Source IP", $" Source Port", $" Destination IP", $" Destination Port",
                            $" Protocol", $" Flow Duration", $" Total Fwd Packets", $" Total Backward Packets",
                            $"Total Length of Fwd Packets", $" Total Length of Bwd Packets", $" Fwd Packet Length Max",
                            $" Fwd Packet Length Min", $" Fwd Packet Length Mean", $" Fwd Packet Length Std", $"Bwd Packet Length Max",
                            $" Bwd Packet Length Min", $" Bwd Packet Length Mean", $" Bwd Packet Length Std", $"Flow Bytes/s",
                            $" Flow Packets/s")
val logregdata = logselect.withColumn("label", when($"label_unreg" === "BENIGN" ,0).otherwise(1)).drop("label_unreg")
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 42)

/// Set up Pipeline

// For the bigger dataset we need to encode...

val sourceIPIndexer = new StringIndexer().setInputCol(" Source IP").setOutputCol("source_ip_index")
val destIPIndexer = new StringIndexer().setInputCol(" Destination IP").setOutputCol("dest_ip_index")
val sourceEncoder = new OneHotEncoder().setInputCol("source_ip_index").setOutputCol("source_ip_vec")
val destEncoder = new OneHotEncoder().setInputCol("dest_ip_index").setOutputCol("dest_ip_vec")
val assembler = new VectorAssembler().setInputCols(Array(
    "source_ip_vec"," Source Port","dest_ip_vec"," Destination Port"," Protocol",
    " Flow Duration"," Total Fwd Packets"," Total Backward Packets","Total Length of Fwd Packets",
    " Total Length of Bwd Packets"," Fwd Packet Length Max"," Fwd Packet Length Min"," Fwd Packet Length Mean",
    " Fwd Packet Length Std","Bwd Packet Length Max"," Bwd Packet Length Min"," Bwd Packet Length Mean",
    " Bwd Packet Length Std","Flow Bytes/s"," Flow Packets/s")).setOutputCol("features")
val lr = new LogisticRegression()

var pipeline = new Pipeline().setStages(Array(
    sourceIPIndexer,
    destIPIndexer,
    sourceEncoder,
    destEncoder,
    assembler,
    lr))

// Split

// Fit
val model = pipeline.fit(training)
model.write.overwrite.save("../../models/canadAI")
// Get Results on Test Set
val results = model.transform(test)

// Need to convert to RDD to use this
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instantiate metrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)
val acc = metrics.accuracy
println("Accuracy:$acc%1.2f")
