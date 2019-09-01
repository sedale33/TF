
require(ramify)
require(R6)

GaussNB <- R6Class("GaussNB", list( K = 0, mu = 0, cov = 0, y = 0,
  fit = function(X, y, epsilon= 1e-2){
    self$y <- y
    self$K <- unique(y)
    data <- cbind(X, y)
    nfeat <- length(data[1,]) - 1
    self$mu <- aggregate(data[,1:nfeat], list(data[,nfeat+1]), mean)
    self$cov <- aggregate(data[,1:nfeat], list(data[,nfeat+1]), function(x) mean((x-mean(x))^2)) + epsilon
    prior <- summary(as.factor(y))/length(y)
    invisible(self)
    },
  perdict = function(X){
    P_hat <- mat.or.vec(length(X[,1]), length(self$K))
    for(i in seq_along(self$K)){
      for(r in seq_along(X[,1])){
        P_hat[r, i] <- sum(-0.5*log(2*pi) - 0.5*log((self$cov[i,-1])) - sum((X[r,]-self$mu[i,-1])^2)/(2*(self$cov[i,-1]^2)))
      }
    }
    argmax(P_hat) - 1
    }
              ))

