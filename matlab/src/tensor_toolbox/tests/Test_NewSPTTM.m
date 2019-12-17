
classdef Test_NewSPTTM < matlab.unittest.TestCase
    
    properties (TestParameter)
        combo = struct('small3d', [3 10 50], 'small4d', [4 10 25], 'small5d', [5 5 10], 'large3d',[3 100 250]);
    end
    
    methods (Test) 

        function Compare(testCase, combo)
            nd = combo(1);
            lsz = combo(2);
            usz = combo(3);
            rsz = usz - lsz;
            sz = lsz * ones(1, nd) + randi(rsz, 1, nd);
            for s1 = 0.1:0.1:1
                X = sptenrand(sz,s1);
                U = cell(nd,1);
                for s2 = 0.1:0.1:1
                    for n = 1:nd
                        U{n} = sprandn(sz(n),lsz + randi(rsz),s2);
                    end
                    for n = 1:nd
                        UD = full(U{n});
                        Y = {ttm(X,U{n},n,'t'),ttm(X,U{n}',n)};
                        YD = {ttm(X,UD,n,'t'), ttm(X,UD', n)};
                        
                        if nnz(Y{1}) > 0.5*prod(size(Y{1}))
                            for example = 1:2
                                testCase.verifyInstanceOf(Y{example},'tensor');
                                testCase.verifyInstanceOf(YD{example},'tensor');
                                testCase.verifyThat(~issparse(Y{example}.data), matlab.unittest.constraints.IsTrue);
                                testCase.verifyThat(~issparse(YD{example}.data), matlab.unittest.constraints.IsTrue);
                            end
                            testCase.verifyEqual(Y{1}.data, Y{2}.data, 'AbsTol', 1e-12);
                            testCase.verifyEqual(Y{1}.data, YD{1}.data, 'AbsTol', 1e-12);
                            testCase.verifyEqual(Y{2}.data, YD{2}.data, 'AbsTol', 1e-12);
                        else
                            for example = 1:2
                                testCase.verifyInstanceOf(Y{example},'sptensor');
                                testCase.verifyInstanceOf(YD{example},'sptensor');
                            end
                            testCase.verifyEqual(size(Y{1}), size(Y{2}));
                            testCase.verifyEqual(size(YD{1}), size(YD{2}));
                            testCase.verifyEqual(size(Y{1}), size(YD{2}));
                            testCase.verifyEqual(size(Y{2}), size(YD{1}));
                            testCase.verifyEqual(Y{1}.subs, Y{2}.subs, 'AbsTol', 1e-12);
                            testCase.verifyEqual(YD{1}.subs, YD{2}.subs, 'AbsTol', 1e-12);
                            testCase.verifyEqual(YD{1}.vals, YD{2}.vals, 'AbsTol', 1e-12);
                            testCase.verifyEqual(Y{1}.vals, Y{2}.vals, 'AbsTol', 1e-12);
                            testCase.verifyEqual(Y{1}.vals, YD{1}.vals, 'AbsTol', 1e-12);
                            testCase.verifyEqual(Y{2}.vals, YD{2}.vals, 'AbsTol', 1e-12);
                        end

                    end
                end
            end
        end
        
    end
end
