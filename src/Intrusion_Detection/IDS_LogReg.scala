
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// stfu
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("data_20p.csv")

val logselect = data.select(data("traffic_decision").as("label_unreg"),
                            $"duration", $"protocol_type", $"service", $"flag", $"src_bytes", $"dst_bytes",
                            $"urgent", $"hot", $"num_failed_logins", $"logged_in", $"root_shell", $"su_attempted",
                            $"num_root", $"num_file_creations", $"num_shells", $"num_access_files", $"num_outbound_cmds",
                            $"is_host_login", $"is_guest_login", $"count", $"srv_count"
                            )
val logregdata = logselect.withColumn("label", when($"label_unreg" === "normal" ,0).otherwise(1)).drop("label_unreg")




// Index and Encode
val protocolIndexer = new StringIndexer().setInputCol("protocol_type").setOutputCol("protocol_type_index")
val serviceIndexer = new StringIndexer().setInputCol("service").setOutputCol("service_index")
val flagIndexer = new StringIndexer().setInputCol("flag").setOutputCol("flag_index")

val protocolEncoder = new OneHotEncoder().setInputCol("protocol_type_index").setOutputCol("protocol_type_vec")
val serviceEncoder = new OneHotEncoder().setInputCol("service_index").setOutputCol("service_vec")
val flagEncoder = new OneHotEncoder().setInputCol("flag_index").setOutputCol("flag_vec")


// Assemble everything together to be ("label","features") format
val assembler = (new VectorAssembler()
                  .setInputCols(Array(
                      "duration", "protocol_type_vec", "service_vec", "flag_vec",
                      "src_bytes", "dst_bytes", "urgent", "hot", "num_failed_logins",
                      "logged_in", "root_shell", "su_attempted", "num_root",
                      "num_file_creations", "num_shells", "num_access_files",
                      "num_outbound_cmds", "is_host_login", "is_guest_login",
                      "count", "srv_count"
                  ))
                  .setOutputCol("features") )



val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 42)

val lr = new LogisticRegression()
var pipeline = new Pipeline().setStages(Array(protocolIndexer,serviceIndexer,flagIndexer,protocolEncoder,flagEncoder,serviceEncoder,assembler,lr))

// Fit the pipeline to training documents.
val model = pipeline.fit(training)

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
