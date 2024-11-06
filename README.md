## Communication-Efficient Sparse Matrix Multiplication Using OpenMPI

This repository contains an implementation of a parallel algorithm for multiplying two sparse matrices A and B, based on the approach detailed in [this article](https://arxiv.org/pdf/2010.08526). The solution uses Open MPI library, designed for high-performance computing.

## Content
* A standard parallel algorithm for sparse matrix multiplication.
* More effective algorithm based on the article, built on top of the standard approach.
* A report benchmarking and comparing the performance of both solutions.

## How to run
You need access to a cluster managed by Slurm. Then use
```
mkdir build; cd build; cmake ..; make
srun ./matmul [-a sparse_matrix_file_a] [-b sparse_matrix_file_b] [-v] [-g g_value] [-t 2D|3D|balanced] [-l value] 
```
where:
* `-a/b sparse_matrix_file_a/b` is a path to [CSR file](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) storing given sparse matrix. The first row contains 4 integers: the number of rows, the number of columns, the total number of non-zero elements, and the maximum number of non-zero elements in each row. The following 3 rows specify values, column indices, and row offsets. Values may be integers or doubles in format 12.345.
* `-v` prints the matrix `C` (the multiplication result) in the row-major order: the first line specifies the number of rows and the number of columns of the result; `i+1`-th line is the `i`-th row of the matrix. This argument is optional.
* `-t type` specifies which version of the algorithm you want to use. Possible type values are 2D and 3D. They refer to the 2D-SUMMA; 3D-SUMMA, respectively (algorithms described in the article). In case of 3D, there will be additional argument (`-l`), the value of layers `l` introduced by the 3D-SUMMA.
* `-l layers` specifies the number of layers in the 3D-SUMMA procedure. Applies only to 3D version of the algorithm and should be ignored in the 2D case.
* `-g g_value` prints the number of elements in `C` greater than the `g_value`.
