import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import count, when, isnull,col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

label_col = "diabetes"

def get_executor_memory(sc):
    executor_memory_status = sc._jsc.sc().getExecutorMemoryStatus()
    executor_memory_status_dict = sc._jvm.scala.collection.JavaConverters.mapAsJavaMapConverter(executor_memory_status).asJava()
    total_used_memory = 0
    for executor, values in executor_memory_status_dict.items():
        total_memory = values._1() / (1024 * 1024)  # Convert bytes to MB
        free_memory = values._2() / (1024 * 1024)    # Convert bytes to MB
        used_memory = total_memory - free_memory
        total_used_memory += used_memory
    return total_used_memory

def draw_graph(total_time, total_RAM, path_name, name = "not_optimal", datanodes = "1"):
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(total_time, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('Time(c)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of time distribution of {name} way for {datanodes} datanodes')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(total_RAM, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('RAM(MB)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of RAM distribution of {name} way for {datanodes} datanodes')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path_name)
    
def full_nan_value_pyspark(dataset):
    return dataset.na.fill(0)

def to_int(dataset):
    columns = ['age', "hypertension", "heart_disease", "blood_glucose_level", "diabetes"]
    for column in columns:
        dataset = dataset.withColumn(column, dataset[column].cast(IntegerType()))
    return dataset

def to_float(dataset):
    columns = ['bmi', "HbA1c_level"]
    for column in columns:
        dataset = dataset.withColumn(column, dataset[column].cast(FloatType()))
    return dataset

def categorical_to_int(dataset):
    categorical_features = ['gender', 'smoking_history']

    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_features]

    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=column + "_encoded") for indexer, column in zip(indexers, categorical_features)]

    pipeline = Pipeline(stages=indexers + encoders)

    model = pipeline.fit(dataset)
    dataset = model.transform(dataset)

    for column in categorical_features:
        dataset = dataset.drop(column)
        dataset = dataset.drop(column + "_index")

    return dataset

def data_preprocess(df):
    df = full_nan_value_pyspark(df)
    df = categorical_to_int(df)
    df = to_int(df)
    df = to_float(df)
    df = df.filter(col("diabetes").isNotNull())
    return df 
    
def train_process(df):
    feature_columns = [el for el in df.columns if el != label_col]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
    
    rf = RandomForestClassifier(featuresCol="features", labelCol=label_col)
    model = rf.fit(train_data)

    predictions = model.transform(test_data)
    return predictions
    
def evaluate(predictions):
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)
    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy on test data: {accuracy}")
    return accuracy