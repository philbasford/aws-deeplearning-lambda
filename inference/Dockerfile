
# Define global args
ARG FUNCTION_DIR="/home/app/"
ARG RUNTIME_VERSION="3.7"
ARG TENSOR_VERSION="tensorflow-training:2.3.1-cpu-py37-ubuntu18.04"
ARG REGION="eu-west-1"
ARG AWS_PUB_ACC="763104351884"

# Stage 1 - bundle base image + runtime
# Grab a fresh copy of the image and install GCC
FROM ${AWS_PUB_ACC}.dkr.ecr.${REGION}.amazonaws.com/${TENSOR_VERSION} as tensorflow-base

# Stage 2 - build function and dependencies
FROM tensorflow-base AS build-image
# Install aws-lambda-cpp build dependencies

ARG FUNCTION_DIR
ARG RUNTIME_VERSION
# Create function directory
RUN mkdir -p ${FUNCTION_DIR}
# Copy handler function
COPY . ${FUNCTION_DIR}
WORKDIR ${FUNCTION_DIR}
# Optional – Install the function's dependencies
RUN python${RUNTIME_VERSION} -m pip install -r requirements.txt --target ${FUNCTION_DIR}
# Install Lambda Runtime Interface Client for Python
RUN python${RUNTIME_VERSION} -m pip install awslambdaric --target ${FUNCTION_DIR}

# Stage 3 - final runtime image
# Grab a fresh copy of the Python image
FROM tensorflow-base
# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}
# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}
# (Optional) Add Lambda Runtime Interface Emulator and use a script in the ENTRYPOINT for simpler local runs
ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
RUN chmod 755 /usr/bin/aws-lambda-rie
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
