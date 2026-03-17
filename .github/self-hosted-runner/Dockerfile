ARG CUDA_VERSION=12.8.0
ARG PYTHON_VERSION=3.14
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
ARG PYTHON_VERSION

RUN apt-get update && apt-get install -y \
    cmake build-essential curl git python3 python3-pip python3-venv jq libicu-dev sudo \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages -r /tmp/requirements.txt \
    && python${PYTHON_VERSION} -m pip install --break-system-packages -r /tmp/requirements.txt

RUN useradd -m -s /bin/bash runner \
    && echo "runner ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/runner

WORKDIR /home/runner/actions-runner

ARG RUNNER_VERSION=2.332.0
RUN curl -fsSL -o runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && tar xzf runner.tar.gz \
    && rm runner.tar.gz \
    && ./bin/installdependencies.sh

RUN chown -R runner:runner /home/runner

COPY entrypoint.sh /home/runner/actions-runner/entrypoint.sh
RUN chmod +x /home/runner/actions-runner/entrypoint.sh

USER runner

ENTRYPOINT ["./entrypoint.sh"]
