cd './RANN'
compile_rann32c;
cd '..'
addpath('./RANN')
check_rann;
sparse_or_dense_tensor(1,0,'./tensor_toolbox');