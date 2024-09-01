#!/usr/bin/env Rscript


# Read arguments passed from Python
args_json <- commandArgs(trailingOnly = TRUE)[1]
# Parse JSON string into a list
args <- jsonlite::fromJSON(args_json)

path_dataframe = args$data
data <- read.csv(file = path_dataframe)
data$time_index <- NULL

score <- args$score
num_slices <- args$num_time_slices
b <- args$num_static_vars

start_up <- function() {
  library(pcalg)
  library(BiDAG)
  library(data.table)
  library(visNetwork)
  library(proxyC)
  library(RcppCNPy)
}
start_up()

dbnscore<-scoreparameters(score,data,dbnpar=list(samestruct = TRUE, slices = num_slices, b = b, stationary = TRUE, rowids = NULL,
                                                          datalist = NULL), DBN=TRUE)
dbnfit<-iterativeMCMC(dbnscore,accum=TRUE,alpha=0.2,plus1it=6,hardlim=10,cpdag=FALSE, alphainit=0.01, scoreout=TRUE)
dbnsamp<-orderMCMC(dbnscore,scoretable = getSpace(dbnfit),MAP=FALSE,chainout=TRUE)
mcmcep<-edgep(dbnsamp)

write.csv(as.matrix(dbnsamp$DAG), file="./results/result.csv")

