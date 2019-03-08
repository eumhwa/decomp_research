
opt_binning = function(X,nbins=10,slack=0.5)
{
  n = nrow(X)
  p = ncol(X)
  
  K = nbins
  N = floor(p/K)
  s = floor(N*slack)
  
  xbar = apply(X,2,mean)
  
  V = seq(1,p,length=K+1) # start position
  V = round(V)
                    
  for(i in 2:K)
  {
    left = max(V[i-1]-V[i]+1,-s)
    intv = seq(left,s,by=1)
    temp = xbar[V[i]+intv]
    indx = which.min(temp)[1]
    V[i] = V[i] + intv[indx]
  }
  
  Z = matrix(0,n,K)
  for(i in 2:K)
  {
    if((V[i]-V[i-1])==1)
      Z[,i-1] = X[,V[i-1]]
    else
      Z[,i-1] = apply(X[,V[i-1]:(V[i]-1)],1,sum)
  }
  if((p-V[K])==1)
      Z[,K] = X[,V[K]]
  else
      Z[,K] = apply(X[,V[K]:p],1,sum)
  
  return(list(dat=Z,index=V))
}

