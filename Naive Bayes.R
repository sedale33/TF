
require(ramify)
require(R6)

GaussNB <- R6Class("GaussNB", list( K = 0, mu = 0, cov = 0,
  fit = function(X, y, epsilon= 1){
    self$K <- unique(y)
    data <- cbind(X, y)
    nfeat <- length(data[1,]) - 1
    self$mu <- aggregate(data[,1:nfeat], list(data[,nfeat+1]), mean)
    self$cov <- aggregate(data[,1:nfeat], list(data[,nfeat+1]), function(x) mean((x-mean(x))^2)) + epsilon
    prior <- summary(as.factor(y))/length(y)
    invisible(self)
    },
  perdict = function(X){
    P_hat <- mat.or.vec(length(y), length(self$K))
    for(i in seq_along(self$K)){
      for(r in seq_along(X[,1])){
        P_hat[r, i] <- sum(-0.5*log(2*pi) - log((self$cov[i,-1])+1) - ((X[r,]-self$mu[i,-1])^2)/(2*(self$cov[i,-1]^2)+1))
      }
    }
    argmax(P_hat) - 1
    }
              ))

