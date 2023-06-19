import boto3
import json

# Specify the SageMaker endpoint name
endpoint_name = 'xgboost-2023-06-13-18-38-30-905'

# Create a SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='eu-north-1')

# Specify the input values for each feature
sepal_length = 15.1
sepal_width = 13.5
petal_length = 100.4
petal_width = 0.2

# Create a list to hold the feature values
feature_values = [sepal_length, sepal_width, petal_length, petal_width]

# Convert the feature values to a CSV string
input_data = ','.join(map(str, feature_values))

# Specify the content type for the input data
content_type = 'text/csv'

# Make predictions using the SageMaker endpoint
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=input_data.encode('utf-8')  # Encode the input data
)

# Parse the prediction response
prediction_result = response['Body'].read().decode('utf-8')
prediction_result = json.loads(prediction_result)

# Print the prediction result
print(prediction_result)