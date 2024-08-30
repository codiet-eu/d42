#!/usr/bin/env Rscript

# Read arguments passed from Python
args_json <- commandArgs(trailingOnly = TRUE)[1]
# Parse JSON string into a list
args <- jsonlite::fromJSON(args_json)

path_dataframe = args$data
data <- read.csv(file = path_dataframe)
data$time_index <- NULL

size <- args$num_time_slices
n <- args$num_vars

start_up <- function() {
  library(dbnR)
  library(bnlearn)
  library(BiDAG)
  library(data.table)
  library(visNetwork)
  library(proxyC)
  library(RcppCNPy)
}
################################
## changing some functions in the original dbnR to avoid errors 
create_blacklist <- function(name, size, acc = NULL, slice = 1){
  # Create the blacklist so that there are no arcs from t to t-1 and within t
  if(size >= slice){
    n <- grep(paste0("t_", (slice-1), "$"), name)
    len <- length(n)
    from <- name[n]
    to = name[-n]
    
    if((size - slice) > 0)
      fromArc <- as.vector(sapply(from, rep, times = (size - slice) * len, simplify=T))
    else
      fromArc = NULL
    
    toArc <- rep(to, times = len)
    
    withinTo <- rep(from, len)
    withinFrom <- as.vector(sapply(from, rep, times = len, simplify = T))
    
    local_blacklist <- cbind(c(withinFrom, fromArc), c(withinTo, toArc))
    
    acc <- create_blacklist(to, size, rbind(acc, local_blacklist), slice + 1)
  }
  
  return(acc)
}

fold_dt_rec <- function(dt, n_prev, size, slice = 1){
  if(size > slice){
    n <- sapply(n_prev,sub, pattern = paste0("_t_", slice-1),
                replacement = paste0("_t_",slice), simplify=T)
    dt[, (n) := shift(.SD, 1), .SDcols = n_prev]
    dt <- dt[-1]
    dt <- fold_dt_rec(dt, n, size, slice + 1)
  }
  
  return (dt)
}
warn_empty_net <- function(obj){
  ret = FALSE
  if(dim(bnlearn::arcs(obj))[1] == 0){
    warning(sprintf("all nodes in %s are independent. The resulting net has no arcs.\n",
                    deparse(substitute(obj))))
    ret = TRUE
  }
  
  return(ret)
}
merge_nets <- function(net0, netCP1, size, acc = NULL, slice = 1){
  if(size > slice){
    within_t = bnlearn::arcs(net0)
    within_t <- apply(within_t, 2, sub, pattern = "_t_0",
                      replacement = paste0("_t_",slice))
    ret <- merge_nets(net0, netCP1, size, rbind(acc,within_t), slice = slice + 1)
  }
  
  else
    ret <- rbind(bnlearn::arcs(net0), acc, bnlearn::arcs(netCP1))
  
  return(ret)
}
dmmhc <- function(dt, size = 2, f_dt = NULL, blacklist = NULL, intra = TRUE,
                  blacklist_tr = NULL, whitelist = NULL, whitelist_tr = NULL, ...){
  
  if(!is.null(dt)){
    dt <- time_rename(dt)
    if(intra){
      dt_copy <- data.table::copy(dt)
      net0 <- bnlearn::rsmax2(x = dt_copy, blacklist = blacklist, whitelist = whitelist, ...) # Static network
    }
  }
  
  if(is.null(f_dt))
    f_dt <- fold_dt_rec(dt, names(dt), size)
  
  blacklist <- create_blacklist(names(f_dt), size)
  blacklist <- rbind(blacklist, blacklist_tr)
  
  net <- bnlearn::rsmax2(x = f_dt, blacklist = blacklist, whitelist = whitelist_tr, ...) # Transition network
  #check_empty_net(net)
  
  if(intra && !warn_empty_net(net0))
    bnlearn::arcs(net) <- merge_nets(net0, net, size)
  class(net) <- c("dbn", class(net))
  
  return(net)
}

learn_dbn_struc <- function(dt, size = 2, method = "dmmhc", f_dt = NULL, ...){
  
  if(!is.null(dt)){
    if(!is.data.table(dt))
      dt <- as.data.table(dt)
  }
  if(!is.null(f_dt))
    initial_folded_dt_check(f_dt)
  
  net <- do.call(method, list(dt = dt, size = size, f_dt = f_dt, ...))
  attr(net, "size") <- size
  
  return(net)
}
dbndata2dbnr<-function(dt,slices) {
  nvar<-ncol(dt)/slices
  dtnew<-matrix(ncol=nvar)
  
  for(i in 1:nrow(dt)) {
    newlinelong<-dt[i,]
    for(j in 1:slices) {
      newline<-newlinelong[1:nvar+nvar*(j-1)]
      dtnew<-rbind(dtnew,newline)
    }
  }
  dtnew<-dtnew[-1,]
  rownames(dtnew)<-1:nrow(dtnew)
  return(as.data.frame(dtnew))
}
######################################

start_up()

df_data<-data.frame(data)
# generate the base_names vector 
base_names <- paste0("V", seq(1, n))
colnames(df_data) <- rep(base_names, size)

datadbnr<-dbndata2dbnr(copy_data,size)

#learning structure with dynamic max min hill climbing (dmmhc) algorithm 
dbnrfit <- learn_dbn_struc(datadbnr, method = "dmmhc") 

dbnrfitmat<-amat(dbnrfit)
dbnrfitmat[1:n,1:n+n]<-dbnrfitmat[1:n+n,1:n]
dbnrfitmat[1:n+n,1:n]<-0

write.csv(as.matrix(dbnrfitmat), file='./results/result.csv', row.names = FALSE) 
