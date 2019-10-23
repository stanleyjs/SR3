# SR3
SR3 Fusion Clustering for multimodal data

## Dependencies (right now):
Python ~3.6
*`pygsp`
*`graphtools`
*`numpy`
*`scipy`
*`pytorch` (no cuda required)

## Instructions (for now): No Julia support in this repo.
 either add the SR3 folder to your path using `sys.path.append` or open a jupyter notebook in this parent directory. Then
```
import SR3
from SR3.math import linalg,solvers,optimization

op = SR3.SR3(solvers.PygspSolver(),k=5,nu=1)
op.fit(X)
`u_vec,v_vecs, sumV = op.getScale(gamma,iter=100)
```
will import the solver, initialize an SR3 instance, and fit the solver to your numpy/tensor/scipy data `X`.  `getScale` solves for your given `gamma::np.ndarray` scales.


## Known Issues:
* Reshapes appear to be wrong for calculating elementwise distances
* The Julia solver requires .so files and is slow
* The Pygsp solver has issues with near-singular matrices
* Make pip wheels and installation documentation + tutorial

