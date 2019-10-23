# SR3
SR3 Fusion Clustering for multimodal data

Dependencies (right now):
Python 3.6
`pygsp`
`graphtools`
`numpy`
`scipy`
`pytorch` (no cuda required)

(For now) To run, either add the SR3 folder to your path using `sys.path.append` or open a jupyter notebook in this parent directory. Then
```import SR3
from SR3.math import linalg,solvers,optimization```

will import the solver,

```op = SR3.SR3(solvers.PygspSolver(),k=5,nu=1)```
will initialize an SR3 solver instance,
`op.fit(X)` will fit the solver to your numpy/tensor/scipy data `X`,
and to get a scale you run 
`u_vec,v_vecs, sumV = op.getScale(gamma,iter=100)`
where `gamma::np.ndarray`.

