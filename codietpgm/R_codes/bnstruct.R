#!/usr/bin/env Rscript

# Read arguments passed from Python
args_json <- commandArgs(trailingOnly = TRUE)[1]
# Parse JSON string into a list
args <- jsonlite::fromJSON(args_json)

path_dataframe = args$data
data <- read.csv(file = path_dataframe)
data$time_index <- NULL

score <- args$score # either 'BIC' or 'AIC'
num_slices <- args$num_time_slices
n <- args$num_vars

start_up <- function() {
  library(bnstruct)
  library(data.table)
  library(visNetwork)
  library(proxyC)
  library(RcppCNPy)
}
start_up()

dataset <- BNDataset(data, rep(FALSE,n), node.sizes=rep(5,n),
                     variables=colnames(data), num.time.steps = num_slices)
bnstructfit <- learn.dynamic.network(dataset, num.time.steps = num_slices, 
                                     scoring.func= score, algo="hc")

write.csv(as.matrix(bnstructfit@dag), file='./results/result.csv') 
