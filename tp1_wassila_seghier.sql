-- Databricks notebook source
-- MAGIC %python
-- MAGIC data = spark.read.csv("/FileStore/tables/2017_fordgobike_tripdata.csv", header="true", inferSchema="true")
-- MAGIC display(data)

-- COMMAND ----------

DROP TABLE IF EXISTS d;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC data.write.format("delta").mode("overwrite").save("/delta/diamonds")

-- COMMAND ----------

DROP TABLE IF EXISTS d;

CREATE TABLE d USING DELTA LOCATION '/delta/diamonds/'

-- COMMAND ----------

SELECT * from d

-- COMMAND ----------

SELECT start_station_name, duration_sec FROM d GROUP BY start_station_name, duration_sec  ORDER BY start_station_name

-- COMMAND ----------

SELECT distinct start_station_name, start_station_latitude, start_station_longitude, user_type from d where start_station_name =="Addison St at Fourth St"

-- COMMAND ----------

select COUNT(distinct start_station_name)  from d

-- COMMAND ----------

select start_station_latitude, start_station_longitude from d

-- COMMAND ----------

select user_type from d

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql import SparkSession
-- MAGIC from pyspark.sql.functions import split
-- MAGIC from pyspark.ml import Pipeline
-- MAGIC from pyspark.ml.classification import DecisionTreeClassifier
-- MAGIC from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
-- MAGIC from pyspark.ml.evaluation import MulticlassClassificationEvaluator
-- MAGIC  
-- MAGIC if __name__ == "__main__":
-- MAGIC      spark = SparkSession\
-- MAGIC          .builder\
-- MAGIC          .appName("Python tp1 ")\
-- MAGIC          .getOrCreate()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pyspark
-- MAGIC data_cb = spark.read.load('/FileStore/tables/2017_fordgobike_tripdata.csv', 
-- MAGIC                            format='csv', 
-- MAGIC                            header='true', 
-- MAGIC                            inferSchema='true')
-- MAGIC type(data_cb)
-- MAGIC 
-- MAGIC 
-- MAGIC 
-- MAGIC 
-- MAGIC Freq = data_cb.groupBy("user_type").count()
-- MAGIC Freq.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC  #Equilibrage des données
-- MAGIC data= data_cb.toPandas()
-- MAGIC data= data.sample(frac=1)
-- MAGIC  
-- MAGIC import seaborn as sns
-- MAGIC from matplotlib import pyplot as plt
-- MAGIC print('Distribution des classes dans l’ensemble de données ')
-- MAGIC print(data['user_type'].value_counts()/len(data))
-- MAGIC sns.countplot('user_type', data=data)
-- MAGIC plt.title('Classes déséquilibrées', fontsize=14)
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC  
-- MAGIC import pandas as pd
-- MAGIC data= data_cb.toPandas()
-- MAGIC data= data.sample(frac=1)
-- MAGIC  # class Suscriber  == 0
-- MAGIC  # class Customer == 1
-- MAGIC  
-- MAGIC d = {'Subscriber':0,'Customer':1}
-- MAGIC data = data.replace(d)
-- MAGIC suscriber = data.loc[data['user_type'] == 1]
-- MAGIC customer = data.loc[data['user_type'] == 0][:110470]
-- MAGIC normal_distributed_df = pd.concat([suscriber, customer])
-- MAGIC new_df = normal_distributed_df.sample(frac=1, random_state=42)
-- MAGIC new_df.shape

-- COMMAND ----------

-- MAGIC %python
-- MAGIC  
-- MAGIC import seaborn as sns
-- MAGIC from matplotlib import pyplot as plt
-- MAGIC print('Distribution des classes dans l’ensemble de données ')
-- MAGIC print(new_df['user_type'].value_counts()/len(new_df))
-- MAGIC sns.countplot('user_type', data=new_df)
-- MAGIC plt.title('Classes réparties de façon égale', fontsize=14)
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.ml import Pipeline
-- MAGIC from pyspark.ml.classification import GBTClassifier
-- MAGIC from pyspark.ml.feature import VectorIndexer, VectorAssembler
-- MAGIC from pyspark.ml.evaluation import BinaryClassificationEvaluator
-- MAGIC from pyspark.ml.linalg import DenseVector
-- MAGIC  
-- MAGIC from pyspark.ml.feature import VectorAssembler
-- MAGIC  
-- MAGIC df = sqlContext.createDataFrame(new_df)
-- MAGIC numericCols = ['duration_sec', 'start_station_latitude', 'start_station_longitude', 'end_station_latitude','end_station_longitude']
-- MAGIC assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
-- MAGIC df = assembler.transform(df)
-- MAGIC df.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC  
-- MAGIC from pyspark.ml.feature import StringIndexer
-- MAGIC  
-- MAGIC label_stringIdx = StringIndexer(inputCol = 'user_type', outputCol = 'labelIndex')
-- MAGIC df = label_stringIdx.fit(df).transform(df)
-- MAGIC df.show()

-- COMMAND ----------

-- MAGIC %python 
-- MAGIC pd.DataFrame(df.take(110), columns=df.columns).transpose()
-- MAGIC 
-- MAGIC 
-- MAGIC train, test = df.randomSplit([0.7, 0.3], seed = 2018)
-- MAGIC print("Training Dataset Count: " + str(train.count()))
-- MAGIC print("Test Dataset Count: " + str(test.count()))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC # RANDOM FOREST
-- MAGIC from pyspark.ml.classification import RandomForestClassifier
-- MAGIC  
-- MAGIC rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'labelIndex')
-- MAGIC rfModel = rf.fit(train)
-- MAGIC predictions = rfModel.transform(test)
-- MAGIC predictions.select('duration_sec', 'start_station_latitude', 'start_station_longitude', 'end_station_latitude','end_station_longitude', 'labelIndex', 'rawPrediction','prediction', 'probability').show(25)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC  
-- MAGIC predictions.select("labelIndex", "prediction").show(10)
-- MAGIC 
-- MAGIC 
-- MAGIC  
-- MAGIC from pyspark.ml.evaluation import MulticlassClassificationEvaluator
-- MAGIC  
-- MAGIC evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
-- MAGIC accuracy = evaluator.evaluate(predictions)
-- MAGIC print("Accuracy = %s" % (accuracy))
-- MAGIC print("Test Error = %s" % (1.0 - accuracy))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC  
-- MAGIC from pyspark.mllib.evaluation import MulticlassMetrics
-- MAGIC from pyspark.sql.types import FloatType
-- MAGIC import pyspark.sql.functions as F
-- MAGIC  
-- MAGIC preds_and_labels = predictions.select(['prediction','labelIndex']).withColumn('labelIndex', F.col('labelIndex').cast(FloatType())).orderBy('prediction')
-- MAGIC preds_and_labels = preds_and_labels.select(['prediction','labelIndex'])
-- MAGIC metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
-- MAGIC print(metrics.confusionMatrix().toArray())

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # DESCISION TREES
-- MAGIC from pyspark.ml.classification import DecisionTreeClassifier
-- MAGIC DT = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'labelIndex')
-- MAGIC model = DT.fit(train)
-- MAGIC predictions = model.transform(test)
-- MAGIC predictions.groupBy("prediction").count().show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC  
-- MAGIC predictions.select("labelIndex", "prediction").show(10)
-- MAGIC 
-- MAGIC 
-- MAGIC  
-- MAGIC from pyspark.ml.evaluation import MulticlassClassificationEvaluator
-- MAGIC  
-- MAGIC evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
-- MAGIC accuracy = evaluator.evaluate(predictions)
-- MAGIC print("Accuracy = %s" % (accuracy))
-- MAGIC print("Test Error = %s" % (1.0 - accuracy))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC  
-- MAGIC from pyspark.mllib.evaluation import MulticlassMetrics
-- MAGIC from pyspark.sql.types import FloatType
-- MAGIC import pyspark.sql.functions as F
-- MAGIC  
-- MAGIC preds_and_labels = predictions.select(['prediction','labelIndex']).withColumn('labelIndex', F.col('labelIndex').cast(FloatType())).orderBy('prediction')
-- MAGIC preds_and_labels = preds_and_labels.select(['prediction','labelIndex'])
-- MAGIC metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
-- MAGIC print(metrics.confusionMatrix().toArray())

-- COMMAND ----------

-- MAGIC %python 
-- MAGIC # GRADIENT BOOSTED TREES
-- MAGIC from pyspark.ml.classification import GBTClassifier
-- MAGIC 
-- MAGIC gbt = GBTClassifier(featuresCol="features",  labelCol = 'labelIndex')
-- MAGIC model = gbt.fit(train)
-- MAGIC predictions = model.transform(test)
-- MAGIC predictions.groupBy("prediction").count().show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC  
-- MAGIC predictions.select("labelIndex", "prediction").show(10)
-- MAGIC 
-- MAGIC 
-- MAGIC  
-- MAGIC from pyspark.ml.evaluation import MulticlassClassificationEvaluator
-- MAGIC  
-- MAGIC evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
-- MAGIC accuracy = evaluator.evaluate(predictions)
-- MAGIC print("Accuracy = %s" % (accuracy))
-- MAGIC print("Test Error = %s" % (1.0 - accuracy))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC  
-- MAGIC from pyspark.mllib.evaluation import MulticlassMetrics
-- MAGIC from pyspark.sql.types import FloatType
-- MAGIC import pyspark.sql.functions as F
-- MAGIC  
-- MAGIC preds_and_labels = predictions.select(['prediction','labelIndex']).withColumn('labelIndex', F.col('labelIndex').cast(FloatType())).orderBy('prediction')
-- MAGIC preds_and_labels = preds_and_labels.select(['prediction','labelIndex'])
-- MAGIC metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
-- MAGIC print(metrics.confusionMatrix().toArray())
