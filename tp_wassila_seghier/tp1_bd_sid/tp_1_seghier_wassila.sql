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
-- MAGIC Freq = data.groupBy("user_type").count()
-- MAGIC Freq.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC data= data.toPandas()
-- MAGIC data= data.sample(frac=1)
-- MAGIC suscriber = data.loc[data['user_type'] == "Suscriber"]
-- MAGIC customer = data.loc[data['user_type'] == "Customer"]
-- MAGIC import seaborn as sns
-- MAGIC from matplotlib import pyplot as plt
-- MAGIC print('Distribution des classes dans l’ensemble de données ')
-- MAGIC print(data['user_type'].value_counts()/len(data))
-- MAGIC sns.countplot('user_type', data=data)
-- MAGIC plt.title('Classes déséquilibrées', fontsize=14)
-- MAGIC plt.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #df = data["user_type"].replace({"Suscriber": "0", "Customer": "1"}, inplace=True)
-- MAGIC #data = data.set_index('user_type')
-- MAGIC data = data.rename(index={'Suscriber':0})
-- MAGIC #data.replace(data["user_type"]=="Suscriber", 0)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC d = {'Subscriber':0,'Customer':1}
-- MAGIC data = data.replace(d)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC print(data)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC data = data.drop(['start_time', 'end_time','start_station_name','end_station_name'], axis=1)
-- MAGIC print(data["user_type"])
-- MAGIC from sklearn.model_selection import train_test_split
-- MAGIC # Separating the independent variables from dependent variables
-- MAGIC X = data.iloc[:,:-1]
-- MAGIC y = data.iloc[:,-1]
-- MAGIC #y=y.astype('int')
-- MAGIC #print(data["user_type"].unique())
-- MAGIC #Split train-test data
-- MAGIC X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC from sklearn.metrics import classification_report, roc_auc_score
-- MAGIC from pyspark.ml.classification import RandomForestClassifier
-- MAGIC 
-- MAGIC 
-- MAGIC Rf = RandomForestClassifier(numTrees=3)
-- MAGIC model = Rf.fit(X_train, y_train)
-- MAGIC pred = model.predict(X_test)
-- MAGIC 
-- MAGIC 
-- MAGIC 
-- MAGIC print("ROC AUC score : ", roc_auc_score(y_test, pred))

-- COMMAND ----------

-- MAGIC 
-- MAGIC %python
-- MAGIC from pyspark.sql import SparkSession
-- MAGIC from pyspark.sql.functions import split
-- MAGIC from pyspark.ml import Pipeline
-- MAGIC from pyspark.ml.classification import DecisionTreeClassifier
-- MAGIC from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
-- MAGIC from pyspark.ml.evaluation import MulticlassClassificationEvaluator
-- MAGIC 
-- MAGIC if __name__ == "__main__":
-- MAGIC     spark = SparkSession\
-- MAGIC         .builder\
-- MAGIC         .appName("Python tp1 ")\
-- MAGIC         .getOrCreate()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import pyspark
-- MAGIC data_cb = spark.read.load('/FileStore/tables/2017_fordgobike_tripdata.csv', 
-- MAGIC                           format='csv', 
-- MAGIC                           header='true', 
-- MAGIC                           inferSchema='true')
-- MAGIC type(data_cb)
-- MAGIC pyspark.sql.dataframe.DataFrame

-- COMMAND ----------

-- MAGIC %python
-- MAGIC Freq = data_cb.groupBy("user_type").count()
-- MAGIC Freq.show()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #Equilibrage des données
-- MAGIC 
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
-- MAGIC # class Suscriber  == 0
-- MAGIC # class Customer == 1
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
-- MAGIC 
-- MAGIC #Machine Learning
-- MAGIC 
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

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC train, test = df.randomSplit([0.7, 0.3], seed = 2018)
-- MAGIC print("Training Dataset Count: " + str(train.count()))
-- MAGIC print("Test Dataset Count: " + str(test.count()))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.ml.classification import RandomForestClassifier
-- MAGIC 
-- MAGIC rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'labelIndex')
-- MAGIC rfModel = rf.fit(train)
-- MAGIC predictions = rfModel.transform(test)
-- MAGIC predictions.select('duration_sec', 'start_station_latitude', 'start_station_longitude', 'end_station_latitude','end_station_longitude', 'labelIndex', 'rawPrediction', 'prediction', 'probability').show(25)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC predictions.select("labelIndex", "prediction").show(10)

-- COMMAND ----------

-- MAGIC %python
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
