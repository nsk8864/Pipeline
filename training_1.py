# training_1.py
import pandas as pd
import numpy as np
import sagemaker
import os
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import boto3
from sagemaker.inputs import TrainingInput

# Set up the SageMaker session and role
sagemaker_session = sagemaker.Session(boto3.Session(region_name='us-east-1'))
role = "arn:aws:iam::657237046012:role/service-role/AmazonSageMaker-ExecutionRole-20230604T121038"

# Load the Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data.insert(0, "target", iris.target)  # Insert target column at the beginning

# Split the data into train, validation, and test sets
train_data, validation_data, test_data = np.split(
    iris_data.sample(frac=1, random_state=1729),
    [int(0.7 * len(iris_data)), int(0.9 * len(iris_data))],
)

# Save train, validation, and test sets as CSV files
train_data.to_csv("train.csv", header=False, index=False)
validation_data.to_csv("validation.csv", header=False, index=False)
test_data.to_csv("test.csv", header=False, index=False)

# Upload train and validation CSV files to S3
bucket = sagemaker_session.default_bucket()
prefix = "iris"
boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "train/train.csv")
).upload_file("train.csv")
boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "validation/validation.csv")
).upload_file("validation.csv")

# Retrieve the SageMaker XGBoost container image URI
container = sagemaker.image_uris.retrieve(
    "xgboost", boto3.Session().region_name, "latest"
)

# Create a SageMaker session
sess = sagemaker.Session()

# Define the input data for training and validation
s3_input_train = TrainingInput(
    s3_data="s3://{}/{}/train".format(bucket, prefix), content_type="csv"
)
s3_input_validation = TrainingInput(
    s3_data="s3://{}/{}/validation/".format(bucket, prefix), content_type="csv"
)

# Create an XGBoost estimator
xgb = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path="s3://{}/{}/output".format(bucket, prefix),
    sagemaker_session=sess,
)

# Set hyperparameters for the XGBoost model
xgb.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    silent=0,
    num_class=3,
    objective="multi:softmax",
    num_round=1,
)

# Train the XGBoost model
xgb.fit({"train": s3_input_train, "validation": s3_input_validation})

# Deploy the trained model as an endpoint
xgb_predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
)
