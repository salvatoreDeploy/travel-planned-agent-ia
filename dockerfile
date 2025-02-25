# Use the AWS base image for Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Install build-essential compiler and tools
RUN microdnf update -y && microdnf install -y gcc-c++ make

# Copy requirements.txt
COPY requirement.txt ${LAMBDA_TASK_ROOT}

# Install packages
RUN pip install -r requirement.txt

# Copy function code
COPY travel-planned-agent-ia.py ${LAMBDA_TASK_ROOT}

# Set the permissions to make the file executable
RUN chmod +x travel-planned-agent-ia.py

# Set the CMD to your handler
CMD ["travel-planned-agent-ia.handler_lambda"]