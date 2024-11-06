#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <fstream>
#include <optional>
#include <vector>


using proc_idx_t = uint32_t;
using matrix_side_t = uint32_t;
using total_elems_t = uint64_t;
using value_t = double;


void check_open(std::ifstream &file);


enum class AlgorithmVersion { two_dim, three_dim, balanced };


// Parse all values given in argv.
class InputData {
private:
    std::string filepath_a, filepath_b;
    bool print_matrix;
    std::optional<value_t> g_value;
    AlgorithmVersion algorithm_version;
    std::optional<proc_idx_t> layers;

public:
    InputData(std::string filepath_a, std::string filepath_b,
        bool print_matrix, std::optional<value_t> g_value,
        AlgorithmVersion algorithm_version, std::optional<int> layers):
        filepath_a(filepath_a),
        filepath_b(filepath_b),
        print_matrix(print_matrix),
        g_value(g_value),
        algorithm_version(algorithm_version),
        layers(layers) {}

    InputData(int argc, char *argv[]);

    bool is_2d();

    bool is_3d();

    bool should_print();

    std::optional<value_t> get_g_value();

    proc_idx_t get_layers();

    std::string get_filepath_a();

    std::string get_filepath_b();
};


class GridIndexer {
private:
    proc_idx_t processes;
    proc_idx_t processes_sqrt;
    proc_idx_t layers;
    matrix_side_t n, m;
    matrix_side_t procs_in_row, procs_in_col;
    matrix_side_t rows_per_proc, cols_per_proc;
    matrix_side_t rows_rest, cols_rest;
    matrix_side_t rows_threshold, cols_threshold;
    std::vector<std::vector<proc_idx_t>> proc_assignment_a, proc_assignment_b;
    std::vector<std::pair<proc_idx_t, proc_idx_t>> proc_coords_a, proc_coords_b;

    proc_idx_t get_1d_idx(matrix_side_t val, matrix_side_t threshold,
        matrix_side_t per_proc, matrix_side_t rest);

public:
    GridIndexer(proc_idx_t processes, proc_idx_t layers, matrix_side_t n);

    proc_idx_t get_num_processes();

    proc_idx_t get_processes_sqrt();

    matrix_side_t get_n();

    proc_idx_t get_layers();

    std::pair<proc_idx_t, proc_idx_t> get_proc_coords_a(proc_idx_t process);

    std::pair<proc_idx_t, proc_idx_t> get_proc_coords_b(proc_idx_t process);

    proc_idx_t get_proc_idx_a(matrix_side_t row, matrix_side_t col);

    proc_idx_t get_proc_idx_b(matrix_side_t row, matrix_side_t col);

    std::pair<proc_idx_t, proc_idx_t> get_proc_columns_c(proc_idx_t process);
};


struct cell_t {
    value_t value;
    matrix_side_t row;
    matrix_side_t col;
};

using matrix_t = std::vector<cell_t>;

// Send a vector of matrix cells via MPI.
void isend_cell_vector(std::vector<cell_t> &to_send, proc_idx_t receiver,
    MPI_Request &request);

// Receive a vector of matrix cells via MPI.
void irecv_cell_vector(std::vector<cell_t> &to_receive, proc_idx_t sender,
    MPI_Request &request);

// Broadcast a vector of matrix cells via MPI.
void ibcast_cell_vector(matrix_t &broadcast, proc_idx_t root, proc_idx_t my_rank,
    MPI_Comm &comm, MPI_Request &request);

// Used to sort cells by row first.
bool row_major_comparator(const cell_t &lhs, const cell_t &rhs);

// Used to sort cells by column first.
bool column_major_comparator(const cell_t &lhs, const cell_t &rhs);

#endif // UTILS_H
