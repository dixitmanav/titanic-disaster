# Build from the repo root:
#   docker build -f src/r/Dockerfile -t titanic-r .
FROM rocker/r-ver:4.5.1

WORKDIR /app

# Install packages first so this layer is cached between code changes
COPY src/r/install_packages.R src/r/install_packages.R
RUN Rscript src/r/install_packages.R

# Copy the code and the (locally downloaded) data
COPY src/r/ src/r/
COPY src/data/ src/data/

CMD ["Rscript", "src/r/titanic_model.R"]
