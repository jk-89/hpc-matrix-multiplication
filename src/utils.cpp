#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <getopt.h>
#include <mpi.h>
#include "utils.h"


void check_open(std::ifstream &file) {
    if (!file.is_open()) {
        std::cerr << "ERROR: File is not open.\n";
        exit(2);
    }
}


InputData::InputData(int argc, char *argv[]) {
    std::string filepath_a, filepath_b;
    bool print_matrix = false;
    std::optional<value_t> g_value;
    AlgorithmVersion algorithm_version = AlgorithmVersion::two_dim;
    std::optional<int> layers;

    int opt;
    while ((opt = getopt(argc, argv, "a:b:vg:t:l:")) != -1) {
        switch (opt) {
            case 'a':
                filepath_a = optarg;
                break;
            case 'b':
                filepath_b = optarg;
                break;
            case 'v':
                print_matrix = true;
                break;
            case 'g':
                g_value = std::stod(optarg);
                break;
            case 'l':
                layers = std::stoi(optarg);
                break;
            case 't':
                if (strcmp(optarg, "2D") == 0) {
                    algorithm_version = AlgorithmVersion::two_dim;
                    break;
                }
                else if (strcmp(optarg, "3D") == 0) {
                    algorithm_version = AlgorithmVersion::three_dim;
                    break;
                }
                else if (strcmp(optarg, "balanced") == 0) {
                    algorithm_version = AlgorithmVersion::balanced;
                    break;
                }
                // Intentional fallthrough to default.
                [[fallthrough]];
            default:
                std::cerr << "Usage: ./matmul [-a sparse_matrix_file_a] "
                    "[-b sparse_matrix_file_b] [-v] [-g g_value] [-t 2D|3D|balanced] "
                    "[-l value]\n";
                exit(2);
        }
    }

    *this = {filepath_a, filepath_b, print_matrix, g_value, algorithm_version, layers};
}

bool InputData::is_2d() {
    return this->algorithm_version == AlgorithmVersion::two_dim;
}

bool InputData::is_3d() {
    return this->algorithm_version == AlgorithmVersion::three_dim;
}

bool InputData::should_print() {
    return this->print_matrix;
}

std::optional<value_t> InputData::get_g_value() {
    return this->g_value;
}

proc_idx_t InputData::get_layers() {
    if (this->layers.has_value())
        return this->layers.value();
    else
        return 1;
}

std::string InputData::get_filepath_a() {
    return this->filepath_a;
}

std::string InputData::get_filepath_b() {
    return this->filepath_b;
}


GridIndexer::GridIndexer(proc_idx_t processes, proc_idx_t layers, matrix_side_t n) {
    auto m = n / layers;
    this->processes = processes;
    this->layers = layers;
    this->n = n;
    this->m = m;
    processes /= layers;
    proc_idx_t processes_sqrt = std::sqrt(processes);
    // Make sure that std::sqrt found proper value.
    while (processes_sqrt * processes_sqrt > processes)
        processes_sqrt--;
    while ((processes_sqrt + 1) * (processes_sqrt + 1) <= processes)
        processes_sqrt++;
    this->processes_sqrt = processes_sqrt;

    this->procs_in_row = processes_sqrt * layers;
    this->procs_in_col = processes_sqrt;
    this->rows_per_proc = n / processes_sqrt;
    this->cols_per_proc = m / processes_sqrt;
    this->rows_rest = n % this->procs_in_col;
    this->cols_rest = n % this->procs_in_row;
    this->rows_threshold = this->rows_rest * (this->rows_per_proc + 1);
    this->cols_threshold = this->cols_rest * (this->cols_per_proc + 1);

    this->proc_coords_a.resize(this->processes);
    this->proc_assignment_a.resize(this->procs_in_col);
    for (proc_idx_t i = 0; i < this->procs_in_col; i++)
        this->proc_assignment_a[i].resize(this->procs_in_row);
    proc_idx_t row = 0, col = 0;
    for (proc_idx_t rest = 0; rest < this->procs_in_col; rest++) {
        for (proc_idx_t i = rest; i < this->processes; i += this->procs_in_col) {
            this->proc_assignment_a[row][col] = i;
            this->proc_coords_a[i] = {row, col};
            row++;
            if (row == this->procs_in_col) {
                row = 0;
                col++;
            }
        }
    }

    this->proc_coords_b.resize(this->processes);
    this->proc_assignment_b.resize(this->procs_in_row);
    for (proc_idx_t i = 0; i < this->procs_in_row; i++)
        this->proc_assignment_b[i].resize(this->procs_in_col);
    proc_idx_t curr_proc = 0;
    for (proc_idx_t layer = 0; layer < this->layers; layer++) {
        for (proc_idx_t i = layer; i < this->procs_in_row; i += this->layers) {
            for (proc_idx_t j = 0; j < this->procs_in_col; j++) {
                this->proc_assignment_b[i][j] = curr_proc;
                this->proc_coords_b[curr_proc] = {i, j};
                curr_proc++;
            }
        }
    }
}

proc_idx_t GridIndexer::get_num_processes() {
    return this->processes;
}

proc_idx_t GridIndexer::get_processes_sqrt() {
    return this->processes_sqrt;
}

matrix_side_t GridIndexer::get_n() {
    return this->n;
}

proc_idx_t GridIndexer::get_layers() {
    return this->layers;
}

std::pair<proc_idx_t, proc_idx_t>
    GridIndexer::get_proc_coords_a(proc_idx_t process) {
    return this->proc_coords_a[process];
}

std::pair<proc_idx_t, proc_idx_t>
    GridIndexer::get_proc_coords_b(proc_idx_t process) {
    return this->proc_coords_b[process];
}

proc_idx_t GridIndexer::get_1d_idx(matrix_side_t val, matrix_side_t threshold,
    matrix_side_t per_proc, matrix_side_t rest) {
    if (threshold > val)
        return val / (per_proc + 1);
    else
        return rest + (val - threshold) / per_proc;
}

proc_idx_t GridIndexer::get_proc_idx_a(matrix_side_t row, matrix_side_t col) {
    auto proc_x = get_1d_idx(row, this->rows_threshold, this->rows_per_proc, this->rows_rest);
    auto proc_y = get_1d_idx(col, this->cols_threshold, this->cols_per_proc, this->cols_rest);
    return this->proc_assignment_a[proc_x][proc_y];
}

proc_idx_t GridIndexer::get_proc_idx_b(matrix_side_t row, matrix_side_t col) {
    auto proc_y = get_1d_idx(col, this->rows_threshold, this->rows_per_proc, this->rows_rest);
    auto proc_x = get_1d_idx(row, this->cols_threshold, this->cols_per_proc, this->cols_rest);
    return this->proc_assignment_b[proc_x][proc_y];
}

std::pair<proc_idx_t, proc_idx_t> GridIndexer::get_proc_columns_c(proc_idx_t process) {
    auto proc_y = process % this->processes_sqrt;
    if (proc_y < this->rows_rest) {
        auto start = proc_y * (this->rows_per_proc + 1);
        return {start, start + this->rows_per_proc + 1};
    }
    else {
        auto start = this->rows_rest * (this->rows_per_proc + 1) +
            (proc_y - this->rows_rest) * (this->rows_per_proc);
        return {start, start + this->rows_per_proc};
    }
}


void isend_cell_vector(matrix_t &to_send, proc_idx_t receiver,
    MPI_Request &request) {
    matrix_side_t count = to_send.size();
    MPI_Send(&count, 1, MPI_UINT32_T, receiver, 0, MPI_COMM_WORLD);
    MPI_Isend(to_send.data(), count * sizeof(cell_t), MPI_BYTE, receiver,
        1, MPI_COMM_WORLD, &request);
}

void irecv_cell_vector(matrix_t &to_receive, proc_idx_t sender,
    MPI_Request &request) {
    matrix_side_t count;
    MPI_Recv(&count, 1, MPI_UINT32_T, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    to_receive.resize(count);
    MPI_Irecv(to_receive.data(), count * sizeof(cell_t), MPI_BYTE, sender,
        1, MPI_COMM_WORLD, &request);
}

void ibcast_cell_vector(matrix_t &matrix, proc_idx_t root, proc_idx_t my_rank,
    MPI_Comm &comm, MPI_Request &request) {
    matrix_side_t count = matrix.size();
    MPI_Bcast(&count, 1, MPI_UINT32_T, (int) root, comm);
    if (my_rank != root)
        matrix.resize(count);
    MPI_Ibcast(matrix.data(), count * sizeof(cell_t), MPI_BYTE, (int) root, comm, &request);
}

bool row_major_comparator(const cell_t &lhs, const cell_t &rhs) {
    if (lhs.row != rhs.row)
        return lhs.row < rhs.row;
    else if (lhs.col != rhs.col)
        return lhs.col < rhs.col;
    else
        return lhs.value < rhs.value;
}

bool column_major_comparator(const cell_t &lhs, const cell_t &rhs) {
    if (lhs.col != rhs.col)
        return lhs.col < rhs.col;
    else if (lhs.row != rhs.row)
        return lhs.row < rhs.row;
    else
        return lhs.value < rhs.value;
}
