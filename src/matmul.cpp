#include <mpi.h>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include "utils.h"


// Go through the file and spread the data to processes.
void spread_input_data(std::ifstream &file, GridIndexer &grid_indexer,
    std::vector<matrix_t> &proc_vals, bool a_matrix) {
    std::string line;
    matrix_side_t matrix_side;
    total_elems_t non_zero_elems, max_non_zero_per_row;
    std::getline(file, line);
    {
        std::istringstream iss(line);
        iss >> matrix_side >> matrix_side >> non_zero_elems >> max_non_zero_per_row;
    }

    std::vector<value_t> vals(non_zero_elems);
    std::vector<matrix_side_t> col_index(non_zero_elems);
    std::vector<matrix_side_t> row_index(matrix_side + 1);

    std::getline(file, line);
    {
        std::istringstream iss(line);
        for (total_elems_t i = 0; i < non_zero_elems; i++)
            iss >> vals[i];
    }
    std::getline(file, line);
    {
        std::istringstream iss(line);
        for (total_elems_t i = 0; i < non_zero_elems; i++)
            iss >> col_index[i];
    }
    std::getline(file, line);
    {
        std::istringstream iss(line);
        for (matrix_side_t i = 0; i <= matrix_side; i++)
            iss >> row_index[i];
    }

    total_elems_t processed = 0;
    for (matrix_side_t row = 0; row < matrix_side; row++) {
        for (total_elems_t j = 0; j < row_index[row + 1] - row_index[row]; j++) {
            matrix_side_t col = col_index[processed];
            proc_idx_t proc_idx;
            if (a_matrix)
                proc_idx = grid_indexer.get_proc_idx_a(row, col);
            else
                proc_idx = grid_indexer.get_proc_idx_b(row, col);
            proc_vals[proc_idx].push_back({vals[processed], row, col});
            processed++;
        }
    }
}


// Process 0 assigns data to all other processes.
void process_0_init_work(InputData &input_data, matrix_t &matrix_a,
    matrix_t & matrix_b, GridIndexer &grid_indexer) {
    std::ifstream a_file(input_data.get_filepath_a());
    check_open(a_file);
    std::ifstream b_file(input_data.get_filepath_b());
    check_open(b_file);

    proc_idx_t num_processes = grid_indexer.get_num_processes();
    std::vector<matrix_t> proc_vals(num_processes);
    MPI_Status status;
    std::vector<MPI_Request> requests_a(num_processes - 1);
    std::vector<MPI_Request> requests_b(num_processes - 1);
    spread_input_data(a_file, grid_indexer, proc_vals, true);
    matrix_a = std::move(proc_vals[0]);
    for (proc_idx_t i = 1; i < num_processes; i++)
        isend_cell_vector(proc_vals[i], i, requests_a[i - 1]);
    for (auto &req : requests_a)
        MPI_Wait(&req, &status);

    proc_vals.clear();
    proc_vals.resize(num_processes);

    spread_input_data(b_file, grid_indexer, proc_vals, false);
    matrix_b = std::move(proc_vals[0]);
    for (proc_idx_t i = 1; i < num_processes; i++)
        isend_cell_vector(proc_vals[i], i, requests_b[i - 1]);
    for (auto &req : requests_b)
        MPI_Wait(&req, &status);

    a_file.close();
    b_file.close();
}


// Assumes that a is sorted row-major, b is sorted column-major.
// Uses sliding window method.
void local_multiply(matrix_t &a, matrix_t &b, matrix_t &c) {
    size_t row_start_a = 0, row_end_a = 1;

    while (row_start_a < a.size()) {
        while (row_end_a != a.size() && a[row_end_a].row == a[row_start_a].row)
            row_end_a++;
        
        size_t col_start_b = 0, col_end_b = 1;
        while (col_start_b < b.size()) {
            while (col_end_b != b.size() && b[col_end_b].col == b[col_start_b].col)
                col_end_b++;

            size_t ptr_a = row_start_a, ptr_b = col_start_b;
            while (ptr_a < row_end_a && ptr_b < col_end_b) {
                if (a[ptr_a].col < b[ptr_b].row) {
                    ptr_a++;
                }
                else if (a[ptr_a].col > b[ptr_b].row) {
                    ptr_b++;
                }
                else {
                    c.push_back({a[ptr_a].value * b[ptr_b].value, a[ptr_a].row, b[ptr_b].col});
                    ptr_a++;
                    ptr_b++;
                }
            }
            
            col_start_b = col_end_b;
        }

        row_start_a = row_end_a;
    }
}

// Calculate the number of values greater than g_value in all processes.
void process_g_value(matrix_t &matrix, value_t g_value, int my_rank) {
    total_elems_t greater = 0;
    for (auto &cell : matrix) {
        if (cell.value > g_value)
            greater++;
    }

    total_elems_t total_greater;
    MPI_Reduce(&greater, &total_greater, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (my_rank == 0)
        std::cout << total_greater << '\n';
}

// Sum up the values from each fixed (row, column) pair.
void merge_matrix_column_major(matrix_t &c) {
    if (c.empty())
        return;

    sort(c.begin(), c.end(), column_major_comparator);
    value_t curr_sum = c[0].value;
    size_t curr_idx = 0;

    for (size_t i = 1; i < c.size(); i++) {
        if (c[i].row != c[i - 1].row || c[i].col != c[i - 1].col) {
            c[curr_idx] = {curr_sum, c[i - 1].row, c[i - 1].col};
            curr_idx++;
            curr_sum = 0;
        }
        curr_sum += c[i].value;
    }
    c[curr_idx] = {curr_sum, c.back().row, c.back().col};
    curr_idx++;
    c.resize(curr_idx);
}

// Gather merged matrices in process 0 and print the whole matrix.
void gather_merged_matrices(matrix_t &matrix, proc_idx_t num_processes,
    matrix_side_t matrix_side, int my_rank) {
    int size = (int) matrix.size() * sizeof(cell_t);
    std::vector<int> sizes(num_processes);
    MPI_Gather(&size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> prefix_offset(num_processes, 0);
    int total_size = 0;
    if (my_rank == 0) {
        for (proc_idx_t i = 0; i < num_processes; i++) {
            prefix_offset[i] = total_size;
            total_size += sizes[i];
        }
    }

    matrix_t whole_matrix;
    if (my_rank == 0)
        whole_matrix.resize(total_size / sizeof(cell_t));
    MPI_Gatherv(matrix.data(), size, MPI_BYTE, whole_matrix.data(),
        sizes.data(), prefix_offset.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        std::sort(whole_matrix.begin(), whole_matrix.end(), row_major_comparator);
        size_t ptr = 0;
        std::cout << matrix_side << ' ' << matrix_side << '\n';
        for (matrix_side_t i = 0; i < matrix_side; i++) {
            for (matrix_side_t j = 0; j < matrix_side; j++) {
                if (ptr != whole_matrix.size() && whole_matrix[ptr].row == i
                    && whole_matrix[ptr].col == j) {
                    std::cout << whole_matrix[ptr].value << ' ';
                    ptr++;
                }
                else {
                    std::cout << "0 ";
                }
            }
            std::cout << '\n';
        }
    }
}

void check_printing_options(InputData &input_data, matrix_t &matrix,
    GridIndexer &grid_indexer, int my_rank) {
    auto g = input_data.get_g_value();
    if (g.has_value())
        process_g_value(matrix, g.value(), my_rank);

    if (input_data.should_print()) {
        gather_merged_matrices(matrix, grid_indexer.get_num_processes(),
            grid_indexer.get_n(), my_rank);
    }
}


void summa2d(GridIndexer &grid_indexer, MPI_Comm &comm_row, MPI_Comm &comm_col,
    int my_rank, matrix_t &matrix_a, matrix_t &matrix_b, matrix_t &matrix_c) {
    // Sort B matrix in a column-major order.
    std::sort(matrix_b.begin(), matrix_b.end(), column_major_comparator);

    MPI_Status status;
    MPI_Request request_a, request_b;
    auto processes_sqrt = grid_indexer.get_processes_sqrt();
    auto [_, proc_y] = grid_indexer.get_proc_coords_a(my_rank);
    auto [proc_x, __] = grid_indexer.get_proc_coords_b(my_rank);
    auto layers = grid_indexer.get_layers();
    proc_x /= layers;
    proc_y /= layers;

    for (proc_idx_t stage = 0; stage < processes_sqrt; stage++) {
        matrix_t recv_a, recv_b;
        // Broadcast via row.
        if (proc_y == stage)
            ibcast_cell_vector(matrix_a, stage, proc_y, comm_row, request_a);
        else
            ibcast_cell_vector(recv_a, stage, proc_y, comm_row, request_a);
        // Broadcast via column.
        if (proc_x == stage)
            ibcast_cell_vector(matrix_b, stage, proc_x, comm_col, request_b);
        else
            ibcast_cell_vector(recv_b, stage, proc_x, comm_col, request_b);

        MPI_Wait(&request_a, &status);
        MPI_Wait(&request_b, &status);
        if (proc_y == stage && proc_x == stage)
            local_multiply(matrix_a, matrix_b, matrix_c);
        else if (proc_y == stage)
            local_multiply(matrix_a, recv_b, matrix_c);
        else if (proc_x == stage)
            local_multiply(recv_a, matrix_b, matrix_c);
        else
            local_multiply(recv_a, recv_b, matrix_c);
    }
}


void all_to_all(matrix_t &matrix_c, matrix_t &final_matrix, std::vector<int> &sendcnt,
    MPI_Comm &comm_fiber, proc_idx_t layers) {
    std::vector<int> senddispl(layers), recvcnt(layers), recvdispl(layers);
    for (auto &it : sendcnt)
        it *= sizeof(cell_t);
    for (proc_idx_t i = 1; i < layers; i++)
        senddispl[i] = senddispl[i - 1] + sendcnt[i - 1];

    // Firstly, fill recvcounts.
    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, comm_fiber);
    int total_recv = 0;
    for (proc_idx_t i = 0; i < layers; i++) {
        recvdispl[i] = total_recv;
        total_recv += recvcnt[i];
    }

    final_matrix.resize(total_recv / sizeof(cell_t));
    MPI_Alltoallv(matrix_c.data(), sendcnt.data(), senddispl.data(), MPI_BYTE,
        final_matrix.data(), recvcnt.data(), recvdispl.data(), MPI_BYTE, comm_fiber);
}


void process(int my_rank, GridIndexer &grid_indexer, InputData &input_data, bool is_2d) {
    matrix_t matrix_a, matrix_b, matrix_c;

    // Process 0 should parse whole input.
    if (my_rank == 0) {
        process_0_init_work(input_data, matrix_a, matrix_b, grid_indexer);
    }
    else {
        MPI_Status status;
        std::vector<MPI_Request> requests(2);
        irecv_cell_vector(matrix_a, 0, requests[0]);
        irecv_cell_vector(matrix_b, 0, requests[1]);
        MPI_Wait(&requests[0], &status);
        MPI_Wait(&requests[1], &status);
    }

    // We introduce two new "worlds" in order to broadcast the data
    // along one row / column in each layer.
    MPI_Comm comm_row, comm_col;
    auto [_, proc_y] = grid_indexer.get_proc_coords_a(my_rank);
    auto [proc_x, __] = grid_indexer.get_proc_coords_b(my_rank);
    MPI_Comm_split(MPI_COMM_WORLD, proc_x, my_rank, &comm_row);
    MPI_Comm_split(MPI_COMM_WORLD, proc_y, my_rank, &comm_col);

    summa2d(grid_indexer, comm_row, comm_col, my_rank, matrix_a, matrix_b, matrix_c);
    merge_matrix_column_major(matrix_c);

    // In 2D (or 3D with 1 layer) setting we don't have to perform ColSplit etc.
    auto layers = grid_indexer.get_layers();
    if (is_2d || (!is_2d && layers == 1)) {
        check_printing_options(input_data, matrix_c, grid_indexer, my_rank);
        return;
    }

    // Perform ColSplit.
    auto [col_start, col_end] = grid_indexer.get_proc_columns_c(my_rank);
    auto cols_in_layer = (col_end - col_start) / layers;
    auto cols_rest = (col_end - col_start) % layers;
    auto cols_threshold = cols_rest * (cols_in_layer + 1);
    std::vector<int> cells_in_proc(layers);
    for (auto &cell : matrix_c) {
        auto col_scaled = cell.col - col_start;
        if (col_scaled < cols_threshold)
            cells_in_proc[col_scaled / (cols_in_layer + 1)]++;
        else
            cells_in_proc[cols_rest + (col_scaled - cols_threshold) / cols_in_layer]++;
    }

    // Perform AllToAll. Introduce new "world" per fiber.
    auto [proc_x_grid, proc_y_grid] = grid_indexer.get_proc_coords_a(my_rank);
    proc_y_grid /= layers;
    auto grid_index = proc_x_grid * grid_indexer.get_processes_sqrt() + proc_y_grid;
    MPI_Comm comm_fiber;
    MPI_Comm_split(MPI_COMM_WORLD, grid_index, my_rank, &comm_fiber);

    matrix_t final_matrix;
    all_to_all(matrix_c, final_matrix, cells_in_proc, comm_fiber, layers);
    merge_matrix_column_major(final_matrix);

    check_printing_options(input_data, final_matrix, grid_indexer, my_rank);
}


int main(int argc, char *argv[]) {
    InputData input_data(argc, argv);
    std::ifstream a_file(input_data.get_filepath_a());
    check_open(a_file);
    std::string line;
    std::getline(a_file, line);
    std::istringstream iss(line);
    // Obtain the size of matrix side from the file.
    matrix_side_t matrix_side;
    iss >> matrix_side;
    a_file.close();

    int num_processes, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    GridIndexer grid_indexer(num_processes, input_data.get_layers(), matrix_side);

    auto processes_sqrt = grid_indexer.get_processes_sqrt();
    auto layers = grid_indexer.get_layers();
    if (layers * processes_sqrt * processes_sqrt != (proc_idx_t) num_processes) {
        if (my_rank == 0)
            std::cerr << "Number of processes is invalid.\n";
        MPI_Finalize();
        exit(1);
    }

    process(my_rank, grid_indexer, input_data, input_data.is_2d());

    MPI_Finalize();

    return 0;
}
