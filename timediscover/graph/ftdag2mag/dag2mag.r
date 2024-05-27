# Functions to accompany Shipley, B. & Douma, J. Douma. 
#Testing piecewise structural equations models in 
#the presence of latent variables, 
#including correlated errors.

## helper functions

#First, make sure that the conditioning latents, if present, are a subset
# of the declared latents.  If not, stop.  
test.conditioning.latents<-function(latents,conditioning.latents){
  if(is.null(conditioning.latents))return(0)
  if(setequal(conditioning.latents,intersect(latents,conditioning.latents)))return(0)
  return(1)
}

#This forms the full conditioning set of observed and conditioning
#latents. The input and output vectors are variable names
full.conditioning.set<-function(observed.conditioning,
                                conditioning.latents){
  union(observed.conditioning,conditioning.latents)
}
# test if there a directed path between two variables that are not d-separated without conditioning. 
is.directed.path<-function(use.dag,start.var,end.var){
  # start.var=character name of first variable in use.dag
  # end.var=character name of second variable in use.dag
  # if there is a directed path between start.var and end.var, and these two variables
  # are not d-separated in use.dag without conditioning, then there is a directed path between them
  # returns as TRUE or FALSE
  var.names<-colnames(use.dag)
  #start.node is a single number giving the column of use.dag containing start.var
  start.node<-(1:length(var.names))[colnames(use.dag)==start.var]
  #end.node is a single number giving the column of use.dag containing end.var
  end.node<-(1:length(var.names))[colnames(use.dag)==end.var]
  #findPath is a function in ggm that finds one path between two nodes of a graph
  #it returns a vector of numbers giving the sequence of nodes of this path
  #starting at st and going to en
  test1<-length(findPath(amat=use.dag,st=start.node,en=end.node))>0
  test2<-!dSep(amat=use.dag,first=start.var,second=end.var,cond=NULL)
  #if TRUE, there is a path from start.var to end.var and this path has no colliders
  test1 & test2
  return(test1 & test2)
}

#pairs.without.edge outputs, as a matrix, the number of pairs of variables in my.graph
# that don't share an edge, with one column
# per pair and the two rows giving the variables in the pair.  
# pairs.without.edge function
pairs.without.edge <- function(my.graph) {
  nvars<-dim(my.graph)[2]
  com <- combn(1:nvars, 2)
  ncombs <- dim(com)[2]
  keep <- rep(T, ncombs)
  for (i in 1:ncombs) {
    # if(there is an edge between this pair) then remove from com
    if (my.graph[com[1, i], com[2, i]] != 0 |
        my.graph[com[2, i], com[1, i]]!=0) {
      com[1, i] <- com[2, i] <- 0
      keep[i]<-F
    }
  }
  matrix(com[, keep],ncol=sum(keep))
}

# find.possible.Q outputs a vector listing all other variables from 1:nvars
# except x and y
find.possible.Q <- function(nvars, x, y) {
  z <- 1:nvars
  z[x] <- z[y] <- 0
  z[z > 0]
}

# converts indices of variables to variable names; for dSep
dag.name<-function (amat,n) 
{
  rownames(amat)[n]
}


#basis Set for MAG
basiSet.mag<-function(cgraph){
  #cgraph has existing edges of 0-->1, 100<-->100 or 10--10
  #mag will have existing edges of 0-->1, 100<-->100 or 10--10

  cat("Basis Set for MAG:","\n")
  cat("I(X,Y|Z) means X is m-separated from Y given the set Z in the MAG",
      "\n")
  mag<-cgraph
  nod<-rownames(mag)
  dv<-length(nod)
  ind<-NULL
  test<-NULL
  for(r in 1:dv){
    for(s in r:dv){
      #if there is an element>0 in column s then r & s are adjacent
      #in mag
      if((mag[r,s]!=0) | (mag[s,r]!=0) | r==s)
        next
      else{
        test<-1
        ed<-nod[c(r,s)]
        pa.r<-nod[mag[,r]==1]
        pa.s<-nod[mag[,s]==1]
        msep<-union(pa.r,pa.s)
        msep<-setdiff(msep,ed)
        b<-list(c(ed,msep))
        ind<-c(ind,b)
        cat("I(",ed[1],",",ed[2],"|",msep,")","\n")
      }
    }
  }
  if(is.null(test))cat("No elements in the basis set","\n")
  return(ind)
}


orient.MAG<-function(full.DAG,latent.conditioners,cgraph,observed.vars){
# RETURNS the oriented cgraph
  
  #This function implements the orientation rules of Richardson & Spirtes
  #full.DAG is the original DAG and latent.conditioners is the names of
  #any latent conditioning variables in it.
  #cgraph is input as a matrix with directed edges (X-->Y),coded as 0-->1,
  #or undirected edges (X--Y), coded as 1--1.
  #This is the matrix after step 3 of Richardson & Spirtes, before the
  #undirected edges are oriented
  #observed.vars is the names of the observed variables in the DAG
  n.observed<-length(observed.vars)
  for(i in 1:n.observed){
    for(j in 1:n.observed){
      # test if there is an undirected edge between variables
      if(cgraph[i,j]==1 & cgraph[j,i]==1){
        test<-0 
        # is i ancestral to j?
        if(is.directed.path(use.dag=full.DAG,start.var=observed.vars[i],
                            end.var=observed.vars[j])){
          # there is a directed path from i to j
          cgraph[i,j]<-1
          cgraph[j,i]<-0
          next
        }
        # is j ancestral to i?
        if(is.directed.path(use.dag=full.DAG,start.var=observed.vars[j],
                            end.var=observed.vars[i])){
          # there is a directed path from i to j
          cgraph[j,i]<-1
          cgraph[i,j]<-0
          next
        }
        # can the edge be turned into correlated errors between i and j?
        if(is.null(latent.conditioners)){
          cgraph[i,j]<-cgraph[j,i]<-100
          next
        }
          
        else
          n.lc<-length(latent.conditioners)
        for(k in 1:n.lc){
          if(is.directed.path(use.dag=full.DAG,start.var=observed.vars[i],
                              end.var= latent.conditioners[k]) &
             is.directed.path(use.dag=full.DAG,start.var=observed.vars[j],
                              
                              end.var=latent.conditioners[k])){
                              test<-1
                              cgraph[i,j]<-cgraph[j,i]<-10 
                              } 
        }
        if(test==0){cgraph[i,j]<-cgraph[j,i]<-100
        }
      }
    }
  }
  cgraph
}


#This is the main function, with several smaller functions defined
#within it.
DAG.to.MAG<-function (full.DAG, latents = NA,
                      conditioning.latents=NULL,
                      verbose=FALSE)
{
  # full.DAG is a binary (0/1) matrix produced from DAG() function in ggm
  # latents is a character vector giving the names of the latents
  # conditioning.latents is a character vector giving the names
  # of those latents that serve as conditioning variables for
  # sampling (i.e. sampling bias)
  # return.graph=TRUE if you want the get the Mixed Acyclic Graph matrix
  # and the basis set of the MAG
  #
  # combn gives all unique combinations of a vector, taken
  # 2 at a time, and outputs a matrix with each unique combination in a
  # column and the total number of columns equal to the total number of
  # unique combinations
  
  #Requires the ggm library 
  library(ggm)
  #First, make sure that the conditioning latents, if present, are a subset
  # of the declared latents.  If not, stop.  
  if(test.conditioning.latents(latents,conditioning.latents)!=0){
    stop("Conditioning latents must be a proper subset of all latents")
  }
  #########################################
  # main function
  #
  full.vars<-row.names(full.DAG)
  full.vars.index<-1:length(full.vars)
  n.observed<-length(full.vars)-length(latents)
  observed.DAG<-full.DAG
  observed.vars<-full.vars
  observed.vars.index<-full.vars.index
  for(i in 1:length(latents)){
    observed.vars[latents[i]==full.vars]<-NA
    observed.vars.index[latents[i]==full.vars]<-NA
    observed.DAG[latents[i]==full.vars,]<-NA
    observed.DAG[,latents[i]==full.vars]<-NA
  }
  latent.vars.index<-match(latents,full.vars)
  
  if(verbose)
      cat("the original DAG is:","\n")
  total.n.vars<-dim(full.DAG)[2]
  if(verbose){
  for(i in 1:(total.n.vars-1)){
    for(j in (i+1):total.n.vars){
      if(full.DAG[i,j]==1 & full.DAG[j,i]==0)cat(full.vars[i],"->",full.vars[j],"\n")
      if(full.DAG[i,j]==0 & full.DAG[j,i]==1)cat(full.vars[j],"->",full.vars[i],"\n")
    }
  }}
  if(sum(is.na(latents))>0){
    return(cat("There are no latents; the DAG doesn't change ","\n"))
  }
  
  if(sum(is.na(latents))==0){
      if(verbose){
        cat("latent variable(s): ",latents,"\n")}
    n.latents<-length(latents)
    for(i in 1:n.latents){
      ok<-F
      for(j in 1:length(full.vars)){
          if(latents[i]==full.vars[j])ok<-T
          }
      if(!ok)return("ERROR: latent variable name not in the DAG")
    }
  }
  if(!is.null(conditioning.latents)){
    if(verbose){
    cat("latents defining implicit conditioning (sampling bias): ",
        conditioning.latents,"\n")}
  }
  if(verbose){
  cat("_____________________","\n")}
  observed.vars<-observed.vars[!is.na(observed.vars)]
  observed.vars.index<-observed.vars.index[!is.na(observed.vars.index)]
  #
  #STEP 2 of Shipley & Douma
  # construct initial observed DAG by removing latents and conserving directed
  # edges between pairs of observed variables
  #
  if(n.observed<=0){return(cat('There is no observed variable.'))}
  if(n.observed==1){return(0)}
  #if(n.observed==2)return(cat("Only two observed variables","\n"))
  
  observed.DAG<-observed.DAG[observed.vars.index,observed.vars.index]
  
  if(n.observed<=0){
    return(cat("All variables are latent; there is no equivalent observed DAG","\n"))
  }
  
  #Finished STEP 2 of Shipley & Douma.
  #Start STEP 3
  pairs.to.test<-pairs.without.edge(observed.DAG)
  n.pairs.to.test<-dim(pairs.to.test)[2]
    n.remaining<-length(observed.vars)-2
  # if all observed variables share an edge then return...
  if(n.pairs.to.test<=0){
  
    return(cat("Since there are only two observed variables, nothing further will be done","\n"))
  }
  add.edge<-matrix(NA,nrow=2,ncol=n.pairs.to.test)
  # for each pair (i) to test, determine dsep in full graph given only the observed variables
  # plus all conditioning latents.
  
  kount<-0
  # i cycles over each pair that are not adjacent...
  for(i in 1:n.pairs.to.test){
    is.pair.dsep<-F
    # get those other observed variables in graph except this pair...
    possible.Q<-find.possible.Q(n.observed,pairs.to.test[1,i],pairs.to.test[2,i])
    
    # Do do unconditional dseparation...
    # i.e. conditional order=0
    first.var<-observed.vars.index[pairs.to.test[1,i]]
    second.var<-observed.vars.index[pairs.to.test[2,i]]
    test<-dSep(amat=full.DAG,first=dag.name(full.DAG,first.var),
               second=dag.name(full.DAG,second.var),
               cond=full.conditioning.set(NULL,
                                          conditioning.latents))
    # if first.var is dsep from second.var then there is no edge between them;
    if(test){
      is.pair.dsep<-T
      
      next
    }
    # if here then there are potential conditional variables to consider
    # so cycle through all possible conditional orders...
    if(sum(is.na(possible.Q)==0)){
      n.possible.Q<-length(possible.Q)
      #now, determine, using observed.vars.index[possible.Q], if the pair are dsep
      # in the full graph
      # j gives the conditional order for a given pair
       for(j in 1:n.possible.Q){
        
        # Q has column = different combinations and rows=elements in each combination
        dQ<-combn(possible.Q,j) 
        
        if(j==n.possible.Q) dQ<-matrix(possible.Q,nrow=j,ncol=1)
        
        n.Q<-dim(dQ)[2]
        
        first.var<-observed.vars.index[pairs.to.test[1,i]]
        
        #   pairs.to.test[1,i],"dag name=",dag.name(full.DAG,first.var),"\n")
        second.var<-observed.vars.index[pairs.to.test[2,i]]
        
        #    pairs.to.test[2,i],"dag.name=",dag.name(full.DAG,second.var),"\n")
        # k cycles through these different combinations
        for(k in 1:n.Q){
          cond.vars<-as.vector(observed.vars.index[dQ[,k]])
          test<-dSep(amat=full.DAG,first=dag.name(full.DAG,first.var),second=dag.name(full.DAG,second.var),
                     cond=full.conditioning.set(dag.name(full.DAG,cond.vars),
                                                conditioning.latents)
                       )
          # if first.var dsep from second.var then there is no edge...
          if(test){
            is.pair.dsep<-T
            break
            }
        }
      }
    }
      if(!is.pair.dsep){
        kount<-kount+1 
        add.edge[1,kount]<-pairs.to.test[1,i]
        add.edge[2,kount]<-pairs.to.test[2,i]
    }
  }
  
  #Add undirected edges to non-adjacent pairs in
  #the observed graph that are not d-separated in the DAG given
  #any combination of other observed variables PLUS any latent conditioning
  #variables.
  
  # convert observed DAG to a partially oriented graph
  cgraph<-matrix(0,n.observed,n.observed,dimnames=list(observed.vars,observed.vars))
  for(i in 1:(n.observed-1)){
    for(j in (i+1):n.observed){
      if(observed.DAG[i,j]==1 & observed.DAG[j,i]==0){
        cgraph[i,j]<-1
        cgraph[j,i]<-0
      }
      if(observed.DAG[j,i]==1 & observed.DAG[i,j]==0){
        cgraph[j,i]<-1
        cgraph[i,j]<-0
      } 
    }
  }
  for(i in 1:kount){
    cgraph[add.edge[1,i],add.edge[2,i]]<-cgraph[add.edge[2,i],add.edge[1,i]]<-1
  } 

  #cgraph now holds the partially oriented inducing graph, with X--Y if there is an inducing
  # path between observed variables X & Y.
  
  #Now, orient these if there are directed paths from i to j
    cgraph<-orient.MAG(full.DAG=full.DAG,latent.conditioners=conditioning.latents,
               cgraph=cgraph,observed.vars=observed.vars)
#
#The next section simply prints the results to the screen
  if(verbose){
      cat("Mixed Acyclic Graph involving only the observed variables:","\n")
      ind.vars<-rep(T,n.observed)
      for(i in 1:(n.observed-1)){
        for(j in (i+1):n.observed){
          if(cgraph[i,j]==1 & cgraph[j,i]==0)cat(observed.vars[i],"->",observed.vars[j],"\n")
          if(cgraph[i,j]==0 & cgraph[j,i]==1)cat(observed.vars[j],"->",observed.vars[i],"\n")
          if(cgraph[i,j]==10 & cgraph[j,i]==10)cat(observed.vars[i],"--",observed.vars[j],"\n")
          if(cgraph[i,j]==100 & cgraph[j,i]==100)cat(observed.vars[i],"<->",observed.vars[j],"\n")
          if(cgraph[i,j]>0)ind.vars[i]<-ind.vars[j]<-FALSE
        }
      }
      if(sum(ind.vars)>0)
        cat("Completely isolated variables",observed.vars[ind.vars],"\n")
      cat("___________________________","\n")
      cat("X->Y means X that is a cause of Y given these observed variables","\n")
      cat("although there could also be latent common causes between them as well.","\n","\n")
      cat("X<->Y means that X and Y are not causes of each other but are correlated","\n")
      cat("via one or more marginal common latent causes.","\n","\n")
      cat("X--Y means that X and Y are not causes of each other but are ","\n")
      cat("correlated via one or more common latent effects that have been conditioned","\n")
      cat("due to biased sampling.","\n","\n")
      cat("___________________________","\n")
      cat("___________________________","\n")}
  return(cgraph)
}



