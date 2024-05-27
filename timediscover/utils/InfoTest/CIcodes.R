#' @importFrom stats dist
NULL
#> NULL


mutualDependenceFree = function(u,v,w)
{
  if(is.null(dim(u))) {n = length(u)} else { n = dim(u)[1] }
  if(is.null(dim(u))) {p = 1} else { p = dim(u)[2] }
  if(is.null(dim(v))) {q = 1} else { q = dim(u)[2] }
  if(is.null(dim(w))) {r = 1} else { r = dim(u)[2] }
  uij = as.matrix(dist(u, diag = T, upper = T))
  ui = exp(-u)+exp(u-1)
  vij = as.matrix(dist(v, diag = T, upper = T))
  vi = exp(-v)+exp(v-1)
  uu = exp(-uij)+ui%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(ui)+2*exp(-1)-4
  vv = exp(-vij)+vi%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(vi)+2*exp(-1)-4
  ww = as.matrix(exp(-dist(w, diag = T, upper = T)))
  conCor = mean(uu*vv*ww)
  conCor
}

mutualDependenceFree_multiZ = function(u,v,w)
{
  if(is.null(dim(u))) {n = length(u)} else { n = dim(u)[1] }
  if(is.null(dim(u))) {p = 1} else {stop("X should be univariate!") }
  if(is.null(dim(v))) {q = 1} else {stop("Y should be univariate!") }
  if(is.null(dim(w))) {stop("Z should be multivariate!")} else { r = dim(w)[2] }
  uij = as.matrix(dist(u, diag = T, upper = T))
  ui = exp(-u)+exp(u-1)
  vij = as.matrix(dist(v, diag = T, upper = T))
  vi = exp(-v)+exp(v-1)
  uu = exp(-uij)+ui%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(ui)+2*exp(-1)-4
  vv = exp(-vij)+vi%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(vi)+2*exp(-1)-4
  ww = as.matrix(exp(-wdist(w)))
  conCor = mean(uu*vv*ww)
  conCor
}

wdist = function(w)
{
  if(is.null(dim(w))) {stop("Z should be multivariate!")} else { r = dim(w)[2] }
  s1 = dist(w[,1], diag = T, upper = T)
  for(i in 2:r)
  {
    s1 = s1+dist(w[,i], diag = T, upper = T)
  }
  s1
}

IndependenceFree = function(u,v)
{
  if(is.null(dim(u))) {n = length(u)} else { n = dim(u)[1] }
  if(is.null(dim(u))) {p = 1} else { p = dim(u)[2] }
  if(is.null(dim(v))) {q = 1} else { q = dim(u)[2] }
  #if(is.null(dim(w))) {r = 1} else { r = dim(u)[2] }
  uij = as.matrix(dist(u, diag = T, upper = T))
  ui = exp(-u)+exp(u-1)
  vij = as.matrix(dist(v, diag = T, upper = T))
  vi = exp(-v)+exp(v-1)
  uu = exp(-uij)+ui%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(ui)+2*exp(-1)-4
  vv = exp(-vij)+vi%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(vi)+2*exp(-1)-4
  #ww = as.matrix(exp(-dist(w, diag = T, upper = T)))
  conCor = mean(uu*vv)
  conCor
}

### Define kernel functions
k_gaussian = function(t, h)
{
  u = t/h 
  return(exp(-t(u)%*%u/2)/sqrt(2*pi)) 
}

k_ep = function(t, h)
{
  u = t/h
  return(0.75*(1-u^2)*(u<=1)*(u>=-1))
}


UEstimate = function(x1, z1, X, Z, h)
{
  #if(is.null(nrow(X))) {n = length(X)} else {n = dim(X)[1,]}
  ZK = GetZK(z1, Z, h)
  uNum = mean(ZK*(X<=x1))
  return(uNum/mean(ZK))
}

VEstimate = function(y1, z1, Y, Z, h)
{
  ZK = GetZK(z1, Z, h)
  vNum = mean(ZK*(Y<=y1))
  return(vNum/mean(ZK))
}

WEstimate = function(z1, Z)
{
  return(mean(Z<=z1))
}


GetZK = function(z1, Z, h)
{
  if(is.null(nrow(Z)))
  {
    n = length(Z)
    Kh = rep(0,n)
    for (i in 1:n) {
      Kh[i] = k_gaussian(z1-Z[i], h)
    }
    return(Kh)
  }
  if(!is.null(nrow(Z)))  {
    n = dim(Z)[1]
    Kh = rep(0,n)
    for (i in 1:n) {
      Kh[i] = k_gaussian(z1-Z[i,], h)
    }
    return(Kh)
  }
}
#' Perform conditional independence test for the variables X and Y conditional on Z, as described in Cai, Z., Li, R., & Zhang, Y. (2022).
#'
#' @param X A univariate variable
#' @param Y A univariate variable
#' @param Z A A univariate variable or multivariate vector
#' @param h kernel bandwidth
#' @return The conditional indepenence measure.
#' @examples
#' n = 100
#' X1 = rnorm(n); X2 = rnorm(n); Z = rnorm(n)
#' X = X1+Z
#' Y = X2+Z
#' h = 1*1.06*sd(Z)*(4/(3*n))^{1/(1+4)}
#' CI.test(X, Y, Z, h)
#' @export
CI.test = function(X, Y, Z, h)
{
  if(is.null(nrow(X))) {n = length(X)} else {n = dim(X)[1,]}
  if(is.null(dim(X))) {p = 1} else {stop("X should be univariate!") }
  if(is.null(dim(Y))) {q = 1} else {stop("Y should be univariate!") }
  if(is.null(dim(Z))) {r = 1} else { r = dim(Z)[2] }

  if(r==1)
  {
    ### Estimate U, V, W
    U = rep(0, n)
    V = U; W = U
    for (i in 1:n) {
      U[i] = UEstimate(X[i], Z[i], X, Z, h)
      V[i] = VEstimate(Y[i], Z[i], Y, Z, h)
      W[i] = WEstimate(Z[i], Z)
    }
    conCor = mutualDependenceFree(U,V,W)
    return(conCor)
  }
  if(r>1)
  {
    #### Estimate U, V, W
    U = rep(0, n)
    V = U; W = matrix(0, n, r)
    for (i in 1:n) {
      U[i] = UEstimate(X[i], Z[i], X, Z, h)
      V[i] = VEstimate(Y[i], Z[i], Y, Z, h)
      for(j in 1:r)
      {
        if(j == 1){ W[i,j] = WEstimate(Z[i,j], Z[,j]) } else { W[i,j] = UEstimate(Z[i,j], Z[i,1:(j-1)], Z[,j], Z[,1:(j-1)],h)}
      }
    }
    conCor = mutualDependenceFree_multiZ(U,V,W)
    return(conCor)
  }
}

#' Perform independence test for the variables X and Y using the method in Cai, Z., Li, R., & Zhang, Y. (2022).
#'
#' @param X A univariate variable
#' @param Y A univariate variable
#' @return The indepenence measure.
#' @examples
#' n = 100
#' X1 = rnorm(n); X2 = rnorm(n); Z = rnorm(n)
#' X = X1+Z
#' Y = X2+Z
#' Independence.test(X, Y)
#' @export
Independence.test = function(X, Y)
{
  if(is.null(nrow(X))) {n = length(X)} else {n = dim(X)[1,]}
  if(is.null(dim(X))) {p = 1} else { p = dim(X)[2] }
  if(is.null(dim(Y))) {q = 1} else { q = dim(Y)[2] }
  #if(is.null(dim(Z))) {r = 1} else { r = dim(Z)[2] }

  #### Estimate U, V,
  U = rep(0, n); V = U;
  for (i in 1:n) {
    U[i] = WEstimate(X[i], X)
    V[i] = WEstimate(Y[i], Y)
  }
  conCor = IndependenceFree(U,V)
  conCor
}

