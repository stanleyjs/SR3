import numpy as np
import torch


class ShrinkageFunction(object):
    def __new__(cls, desired_base, epsilon=1e-8):
        x = type(desired_base.__name__ + 'Shrinkage',
                 (ShrinkageFunction, desired_base), {'epsilon': epsilon})
        return super(ShrinkageFunction, cls).__new__(x)

    def exact(self, theta, gamma, total=False):
        if total:
            return self.exact_total(theta, gamma)
        else:
            return self._exact(theta, gamma, total=False)

    def exact_total(self, theta, gamma, total=True):
        return self._exact(theta, gamma, total=True)

    def prox(self, theta, gamma):
        self._parse_theta(theta)
        self._prox(gamma)

    def _parse_theta(self, theta):
        self.sz = sz = theta.shape
        self.theta = theta.reshape(1, -1)
        self.edges = sz[0]
        self.sgn = np.sign(theta)
        self.theta = np.abs(theta)
        return self.sz, self.theta, self.edges

class Log(object):
    def __init__(self, super):
        print(self)

    def _prox(self, gamma):
        print(gamma)

    def _exact(self, theta, gamma, total=True):
        if total:
            return torch.sum(torch.log(torch.abs(theta + self.epsilon)))
        else:
            return torch.log(torch.abs(theta + self.epsilon))

class Snowflake(object):
    def __init__(self, super):
        pass

    def _prox(self, gamma):
        pass

    def _exact(self, theta, gamma, total=True):
        pass

#### The following is a port of Julia code from Dan Spielman
"""
function sddmWrapLap(lapSolver, sddm::AbstractArray; tol::Real=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[], params...)

    # Make a new adj matrix, a1, with an extra entry at the end.
    a, d = adj(sddm)
    a1 = extendMatrix(a,d)
    F = lapSolver(a1; tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts, params...)

    # make a function that solves the extended system, modulo the last entry
    tol_=tol
    maxits_=maxits
    maxtime_=maxtime
    verbose_=verbose
    pcgIts_=pcgIts

    f = function(b; tol=tol_, maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_)
        xaug = F([b; -sum(b)], tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts)
        xaug = xaug .- xaug[end]
        return xaug[1:a.n]
    end

    return f

end

function sddmWrapLap(lapSolver)
    f = function(sddm::AbstractArray; tol::Real=1e-6, maxits=Inf, maxtime=Inf, verbose=false, pcgIts=Int[], params...)
        return sddmWrapLap(lapSolver, sddm;  tol=tol, maxits=maxits, maxtime=maxtime, verbose=verbose, pcgIts=pcgIts, params... )
    end
    return f
end
"""

"""
    solver = approxchol_lap(a); x = solver(b);
    solver = approxchol_lap(a; tol::Real=1e-6, maxits=1000, maxtime=Inf, verbose=false, pcgIts=Int[], params=ApproxCholParams())
A heuristic by Daniel Spielman inspired by the linear system solver in https://arxiv.org/abs/1605.02353 by Rasmus Kyng and Sushant Sachdeva.  Whereas that paper eliminates vertices one at a time, this eliminates edges one at a time.  It is probably possible to analyze it.
The `ApproxCholParams` let you choose one of three orderings to perform the elimination.
* ApproxCholParams(:given) - in the order given.
    This is the fastest for construction the preconditioner, but the slowest solve.
* ApproxCholParams(:deg) - always eliminate the node of lowest degree.
    This is the slowest build, but the fastest solve.
* ApproxCholParams(:wdeg) - go by a perturbed order of wted degree.
For more info, see http://danspielman.github.io/Laplacians.jl/latest/usingSolvers/index.html

function approxchol_lap(a::SparseMatrixCSC{Tv,Ti};
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams()) where {Tv,Ti}

  if minimum(a.nzval) < 0
      error("Adjacency matrix can not have negative edge weights")
  end

    return Laplacians.lapWrapComponents(approxchol_lap1, a,
    verbose=verbose,
    tol=tol,
    maxits=maxits,
    maxtime=maxtime,
    pcgIts=pcgIts,
    params=params)


end

function approxchol_lapGreedy(a::SparseMatrixCSC;
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams())

  tol_ =tol
  maxits_ =maxits
  maxtime_ =maxtime
  verbose_ =verbose
  pcgIts_ =pcgIts

  t1 = time()

  la = lap(a) # a hit !?

  llmat = LLmatp(a)
  ldli = approxChol(llmat)
  F(b) = LDLsolver(ldli, b)

  if verbose
    println("Using greedy degree ordering. Factorization time: ", time()-t1)
    println("Ratio of operator edges to original edges: ", 2 * length(ldli.fval) / nnz(a))
  end

  if verbose
      println("ratio of max to min diagonal of laplacian : ", maximum(diag(la))/minimum(diag(la)))
  end


  f(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) = pcg(la, b .- mean(b), F, tol=tol, maxits=maxits, maxtime=maxtime, pcgIts=pcgIts, verbose=verbose, stag_test = params.stag_test)

end

function approxchol_lapGiven(a::SparseMatrixCSC;
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams())

  tol_ =tol
  maxits_ =maxits
  maxtime_ =maxtime
  verbose_ =verbose
  pcgIts_ =pcgIts

  t1 = time()

  la = lap(a)

  llmat = LLMatOrd(a)
  ldli = approxChol(llmat)
  F(b) = LDLsolver(ldli, b)

  if verbose
    println("Using given ordering. Factorization time: ", time()-t1)
    println("Ratio of operator edges to original edges: ", 2 * length(ldli.fval) / nnz(a))
  end

  if verbose
      println("ratio of max to min diagonal of laplacian : ", maximum(diag(la))/minimum(diag(la)))
  end


  f(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) = pcg(la, b .- mean(b), F, tol=tol, maxits=maxits, maxtime=maxtime, pcgIts=pcgIts, verbose=verbose, stag_test = params.stag_test)

end

function approxchol_lapWdeg(a::SparseMatrixCSC;
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams())

  tol_ =tol
  maxits_ =maxits
  maxtime_ =maxtime
  verbose_ =verbose
  pcgIts_ =pcgIts

  t1 = time()

  la = lap(a)

  v = vec(sum(a,dims=1))
  v = v .* (1 .+ rand(length(v)))
  p = sortperm(v)

  llmat = LLMatOrd(a,p)
  ldli = approxChol(llmat)

  ip = invperm(p)
  ldlip = LDLinv(p[ldli.col], ldli.colptr, p[ldli.rowval], ldli.fval, ldli.d[ip]);

  F = function(b)
    x = zeros(size(b))
    x = LDLsolver(ldlip, b)
    #x[p] = LDLsolver(ldli, b[p])
    return x
  end

  if verbose
    println("Using wted degree ordering. Factorization time: ", time()-t1)
    println("Ratio of operator edges to original edges: ", 2 * length(ldli.fval) / nnz(a))
  end

  if verbose
      println("ratio of max to min diagonal of laplacian : ", maximum(diag(la))/minimum(diag(la)))
  end


  f(b;tol=tol_,maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_) = pcg(la, b .- mean(b), F, tol=tol, maxits=maxits, maxtime=maxtime, pcgIts=pcgIts, verbose=verbose, stag_test = params.stag_test)

end



function approxchol_lap1(a::SparseMatrixCSC{Tv,Ti};
  tol::Real=1e-6,
  maxits=1000,
  maxtime=Inf,
  verbose=false,
  pcgIts=Int[],
  params=ApproxCholParams()) where {Tv,Ti}

    tol_ =tol
    maxits_ =maxits
    maxtime_ =maxtime
    verbose_ =verbose
    pcgIts_ =pcgIts


    if params.order == :deg

      return approxchol_lapGreedy(a,
        verbose=verbose,
        tol=tol,
        maxits=maxits,
        maxtime=maxtime,
        pcgIts=pcgIts,
        params=params)


    elseif params.order == :wdeg

      return approxchol_lapWdeg(a,
        verbose=verbose,
        tol=tol,
        maxits=maxits,
        maxtime=maxtime,
        pcgIts=pcgIts,
        params=params)


    else
      return approxchol_lapGiven(a,
        verbose=verbose,
        tol=tol,
        maxits=maxits,
        maxtime=maxtime,
        pcgIts=pcgIts,
        params=params)


    end

end
"""