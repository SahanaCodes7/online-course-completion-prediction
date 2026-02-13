FROM public.ecr.aws/lambda/python:3.10

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy function code
COPY lambda_retraining.py .

# Set Lambda handler
CMD ["lambda_retraining.lambda_handler"]
