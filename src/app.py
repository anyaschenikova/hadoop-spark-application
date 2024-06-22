import argparse
import time
import os
import numpy as np
from tqdm import tqdm

from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel

from utils import *

def whole_train_process_to_get_accuracy(df):
    df = data_preprocess(df)
    predictions = train_process(df)
    accuracy = evaluate(predictions)
    return accuracy

def main(data_path, datanodes):
    spark = SparkSession.builder \
        .appName("RegularSparkApp") \
        .getOrCreate()

    sc = spark.sparkContext
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)

    total_time = []
    total_RAM = []

    for i in tqdm(range(100)):
        start_time = time.time()
        
        df = spark.read.csv(data_path, header=True)
        
        accuracy = whole_train_process_to_get_accuracy(df)
        
        end_time = time.time()
        total_time.append(end_time - start_time)
        total_RAM.append(get_executor_memory(sc))

    draw_graph(total_time, total_RAM, f'./not_optimal_with_{datanodes}_datanodes.png', datanodes)

    print()
    print('Average memory(MB):', np.mean(total_RAM))
    print('Average time(c):', np.mean(total_time))
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()          # positional argument
    parser.add_argument("--data-path")
    parser.add_argument("--datanodes", choices=["1", "3"], default = "1") # optimal or not
    
    args = parser.parse_args()
    main(args.data_path, args.datanodes)