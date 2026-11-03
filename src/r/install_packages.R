# Installs the R packages used by titanic_model.R.
# rocker/r-ver:4.5.1 points at a dated CRAN snapshot, so the package
# versions installed here are fixed by the image tag in the Dockerfile.
packages <- c("dplyr")

install.packages(packages, repos = getOption("repos"))

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Failed to install package: ", pkg)
  }
  cat("Installed", pkg, as.character(packageVersion(pkg)), "\n")
}
