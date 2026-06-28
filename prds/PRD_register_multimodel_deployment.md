# Product Requirements Document

## Description
I need to design a solution that effeciently logs mlflow runs in a structure of parent and child 
runs. The main benchmark is to be able to log 500k in 10 minutes at best, and at worst 20 
minutes. The model training to simulate is a multinode cpu training job that trains 500k models 
in parallel in a pandas udf in a spark cluster. Each pandas udf worker node, returns back a 
dataframe with the model details (model name, model as binary string, etc etc).

There are three main approaches of how you can do this. 

## Approach 1: One experiment - One artifact table - One model

After multinode training is complete. Return all models to a single dataframe on the driver 
node. Log a table that holds all 500k models with metadata (model_name, model_as_binary_string, 
etc etc). 

## Approach 2: Log 500k to 1 experiment with nested runs of target heirarchy
After multinode training is complete. Return all models details to a single dataframe on the
driver. Subsequently iterate through the dataframe to log the desired heirarchy of models. For
example, training a demand forecast at the region + sku level, you can log a parent run for each
region, and then log child runs for each sku under the parent region run, and then another level of child runs for each model trained for the sku.

```mermaid
graph TD
    A[Experiment] --> B[Region 1]
    A --> C[Region 2]
    A --> D[Region N]

    B --> E[SKU 1]
    B --> F[SKU 2]
    B --> G[SKU M]

    E --> H[Model 1]
    E --> I[Model 2]
    E --> J[Model 3]

    F --> K[Model 1]
    F --> L[Model 2]
    F --> M[Model 3]
```

## Approach 3: Distribute logging across n experiments with nested runs of target heirarchy

```mermaid
graph TD
    A[Experiment 1] --> B1[Region 1]
    A --> B2[Region 2]

    B1 --> C1[SKU 1]
    B1 --> C2[SKU 2]

    C1 --> D1[Model 1]
    C1 --> D2[Model 2]
    C1 --> D3[Model 3]

    E[Experiment 2] --> F1[Region 3]
    E --> F2[Region 4]

    F1 --> G1[SKU 3]
    F1 --> G2[SKU 4]

    G1 --> H1[Model 1]
    G1 --> H2[Model 2]
    G1 --> H3[Model 3]

    I[Experiment N] --> J1[Region N-1]
    I --> J2[Region N]

    J1 --> K1[SKU M-1]
    J1 --> K2[SKU M]

    K1 --> L1[Model 1]
    K1 --> L2[Model 2]
    K1 --> L3[Model 3]
```
Each Region is mapped to an experiment. Where the experiment name should be the region name with 
prefix with `Demand_Forecasting-[REGION]`.

# Best Practices for Fast MLFlow Logging at Scale

- Refer to [MLFlow Logging Best Practice](../docs/MLflow%20Logging%20at%20Scale%20%E2%80%94%20Best%20Practices%20(Albertsons).pdf)
