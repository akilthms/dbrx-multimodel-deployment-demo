# Large Scale Registration of multi-model forecasting

## Problem Statement
A large retail customer needs to log 500k models across the heirarchy of region -> sku -> ml 
model.However, it is not obvious to customer how to do this efficiently and the best practices 
in mind. The objective of this solution is to coalesce best practices of how to log mlflow runs 
at massive scale. 


## Description
I need to design a solution that efficiently logs mlflow runs in a structure of nested runs in a 
heirarchal structure; Experiment is mapped to the demand forecasting across all the skus for a 
given region. Each region trains a model for each sku. For each sku we train three models 
(AutoArima, Prophet, Random Forest) . The main 
benchmark is to be able to log 500k in 10 minutes at best, and at worst 20 
minutes. 

The demo has effectively three portions to it:
- Generate Synthetic Training Dataset
- Simulate Model Training using spark and pandas udf
- Collect the output from each pandas_udf worker and collect back to driver. Multi-thread the 
  logging for each model found in the returned dataframe on the driver to the target heirarchy 
  (region (experiment)) -> sku -> model). 

Follow the approach of Approach 3 in [Logging Architecture Approaches](./PRD_register_multimodel_deployment.md)


### Model Training
Simulate a multimodel run on a cluster with two worker nodes. Note the objective is to log the 
runs and not to actually train the model. Therefore based on a synthetic dataset.
Simulate 500k model training runs 
by returning back a pandas dataframe with the following data schemas. You may train one model on 
the synthetic data and then use that one model to generate the output. Again, the main objective 
is to log runs at large scale, and not focusing on the quality of the model. 

##### Data Schema - of Training Dataset
Here's a 7-column dataset for demand forecasting:

| Column | Data Type | Description |
|--------|-----------|-------------|
| **date** | datetime/date | The date of the observation |
| **product_id** | string/int | Product identifier |
| **demand** | integer | Number of units sold (target variable) |
| **price** | float | Product price on that date |
| **day_of_week** | integer | Day of week (0-6 or 1-7) |
| **promotion** | boolean/int | Whether a promotion was active (1/0) |
| **inventory** | integer | Available stock/inventory level |

Use the python library `faker` to generate the dataset.

```python
from dataclasses import dataclass, field
from datetime import datetime
from faker import Faker
import random
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, \
    BooleanType, DateType


# NOTE: This dataclass should be moved to entities.py
@dataclass
class DemandRecord:
    date: datetime = field(
        default_factory=lambda: fake.date_between(start_date='-1y', end_date='today'))
    product_id: str = field(default_factory=lambda: f"PROD_{random.randint(1, 50):03d}")
    demand: int = field(default_factory=lambda: random.randint(10, 500))
    price: float = field(default_factory=lambda: round(random.uniform(9.99, 299.99), 2))
    day_of_week: int = field(init=False)
    promotion: bool = field(default_factory=lambda: fake.boolean(chance_of_getting_true=30))
    inventory: int = field(default_factory=lambda: random.randint(50, 1000))

    def __post_init__(self):
        self.day_of_week = self.date.weekday()


class SyntheticDataFactory:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.schema = StructType([
            StructField("date", DateType(), True),
            StructField("product_id", StringType(), True),
            StructField("demand", IntegerType(), True),
            StructField("price", FloatType(), True),
            StructField("day_of_week", IntegerType(), True),
            StructField("promotion", BooleanType(), True),
            StructField("inventory", IntegerType(), True)
        ])

    def generate_dataframe(self, n: int = 1000) -> DataFrame:
        """Generate a synthetic demand forecasting dataset of size n."""
        # Generate n records
        records = [DemandRecord() for _ in range(n)]

        # Convert to list of tuples for PySpark
        data = [(r.date, r.product_id, r.demand, r.price, r.day_of_week, r.promotion, r.inventory)
                for r in records]

        # Create DataFrame
        return self.spark.createDataFrame(data, schema=self.schema)


# Initialize Faker and Spark
fake = Faker()
spark = SparkSession.builder.appName("DemandForecast").getOrCreate()

# Usage
factory = SyntheticDataFactory(spark)
df = factory.generate_dataframe(n=1000)
df.show(10)

```

#### Data Schema - of Returned Panadas DataFrame from training run

| Column       | Data Type     | Description                                                                       |
|--------------|---------------|-----------------------------------------------------------------------------------|
| Region       | string        | One of 5 regions in North America                                                 |
| Sku          | string (uuid) | Product SKU identifier from a unique list of 100k UUIDs shared across all regions |
| model_name   | string        | Model type: AutoArima, Prophet, or XGBoost                                        |
| model_string | binary        | Serialized model object                                                           |


### Best Practices for MLFlow Logging
Refer to this document to incorporate best practices. [MLFlow Logging Best Practice](../docs/MLflow%20Logging%20at%20Scale%20%E2%80%94%20Best%20Practices%20(Albertsons).pdf)

The logging should be async multithreading from the return dataframe on the driver node that 
should have the ability to configure the level of 
concurrency. 