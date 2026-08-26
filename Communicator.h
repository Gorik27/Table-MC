#pragma once

#include <mpi.h>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <type_traits>

template <typename T>
MPI_Datatype get_mpi_type() {
    if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, float>)  return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, int>)    return MPI_INT;
    else {
        throw std::runtime_error("Указанный тип данных не поддерживается MPI");
    }
}

// Теперь для инициализации нужен только ключ словаря
struct InitRequest {
    int key_id; 
};

template <typename T>
class MatrixExchanger {
private:
    int m_rank;
    int m_size;

    struct Peer {
        int rank;
        int local_matrix_rows; // Сколько строк в НАШЕЙ матрице (нужно для этого соседа)
        int peer_matrix_rows;  // Сколько строк в матрице СОСЕДА (размер столбца, который мы получим)
        
        std::vector<T> send_buffer; 
        std::vector<T> recv_buffer;
        
        // Храним индексы столбцов J нашей матрицы, которые у нас запросил этот сосед
        std::vector<int> requested_columns;

        MPI_Request send_msg = MPI_REQUEST_NULL;
        MPI_Request recv_msg = MPI_REQUEST_NULL;
    };

    std::vector<Peer> m_peers;
    std::vector<MPI_Request> m_loop_requests;
    MPI_Datatype m_mpi_type;

public:
    MatrixExchanger(int rank, int size) : m_rank(rank), m_size(size) {
        m_mpi_type = get_mpi_type<T>(); // Функция get_mpi_type<T>() остается прежней
    }

    ~MatrixExchanger() {
        for (auto& peer : m_peers) {
            if (peer.send_msg != MPI_REQUEST_NULL) MPI_Request_free(&peer.send_msg);
            if (peer.recv_msg != MPI_REQUEST_NULL) MPI_Request_free(&peer.recv_msg);
        }
    }

    // --- ИНИЦИАЛИЗАЦИЯ (Абсолютно безопасная версия без дедлоков) ---
void initialize_connections(
    const Matrix<T>& my_matrix,                                 
    const std::vector<int>& target_ranks,                       
    const std::vector<std::vector<InitRequest>>& my_requests,   
    const std::unordered_map<int, int>& local_dict              
) {
    int num_targets = target_ranks.size();
    m_peers.resize(num_targets);

    MPI_Datatype MPI_INIT_REQ;
    MPI_Type_contiguous(1, MPI_INT, &MPI_INIT_REQ); // В InitRequest теперь только 1 int (key_id)
    MPI_Type_commit(&MPI_INIT_REQ);

    // Массивы для метаданных: [кол-во_столбцов, наши_строки]
    std::vector<std::array<int, 2>> send_meta(num_targets);
    std::vector<std::array<int, 2>> recv_meta(num_targets);
    
    std::vector<MPI_Request> meta_requests;
    meta_requests.reserve(num_targets * 2);

    // =========================================================================
    // ШАГ 1: БЕЗОПАСНЫЙ ОБМЕН МЕТАДАННЫМИ
    // =========================================================================
    for (int i = 0; i < num_targets; ++i) {
        m_peers[i].rank = target_ranks[i];
        m_peers[i].local_matrix_rows = my_matrix.rows();

        send_meta[i][0] = static_cast<int>(my_requests[i].size());
        send_meta[i][1] = my_matrix.rows();

        // Сначала выставляем неблокирующий прием от соседа (тег 10)
        MPI_Request r_req;
        MPI_Irecv(recv_meta[i].data(), 2, MPI_INT, target_ranks[i], 10, MPI_COMM_WORLD, &r_req);
        meta_requests.push_back(r_req);

        // Затем инициируем отправку соседу (тег 10)
        MPI_Request s_req;
        MPI_Isend(send_meta[i].data(), 2, MPI_INT, target_ranks[i], 10, MPI_COMM_WORLD, &s_req);
        meta_requests.push_back(s_req);
    }

    // Ждем окончания обмена метаданными
    MPI_Waitall(meta_requests.size(), meta_requests.data(), MPI_STATUSES_IGNORE);
    meta_requests.clear();

    // Выделяем буферы на основе полученных метаданных
    std::vector<int> recv_counts(num_targets);
    for (int i = 0; i < num_targets; ++i) {
        recv_counts[i] = recv_meta[i][0];
        m_peers[i].peer_matrix_rows = recv_meta[i][1];

        m_peers[i].recv_buffer.resize(my_requests[i].size() * m_peers[i].peer_matrix_rows);
        m_peers[i].send_buffer.resize(recv_counts[i] * my_matrix.rows());
    }

    // =========================================================================
    // ШАГ 2: БЕЗОПАСНЫЙ ОБМЕН КЛЮЧАМИ ЗАПРОСОВ
    // =========================================================================
    std::vector<std::vector<InitRequest>> incoming_requests(num_targets);
    std::vector<MPI_Request> req_requests;
    req_requests.reserve(num_targets * 2);

    for (int i = 0; i < num_targets; ++i) {
        incoming_requests[i].resize(recv_counts[i]);

        // Прием ключей (тег 20)
        MPI_Request r_req;
        MPI_Irecv(incoming_requests[i].data(), recv_counts[i], MPI_INIT_REQ, target_ranks[i], 20, MPI_COMM_WORLD, &r_req);
        req_requests.push_back(r_req);

        // Отправка ключей (тег 20)
        MPI_Request s_req;
        MPI_Isend(const_cast<InitRequest*>(my_requests[i].data()), my_requests[i].size(), MPI_INIT_REQ, target_ranks[i], 20, MPI_COMM_WORLD, &s_req);
        req_requests.push_back(s_req);
    }

    // Ждем окончания обмена ключами
    MPI_Waitall(req_requests.size(), req_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Type_free(&MPI_INIT_REQ);

    // =========================================================================
    // ШАГ 3: ЛОКАЛЬНЫЙ ПЕРЕВОД И НАСТРОЙКА КАНАЛОВ (Без изменений)
    // =========================================================================
    for (int i = 0; i < num_targets; ++i) {
        m_peers[i].requested_columns.reserve(recv_counts[i]);
        for (const auto& req : incoming_requests[i]) {
            // ОПАСНОСТЬ ЗАВИСАНИЯ ТУТ: Если ключа нет в словаре, программа выбросит std::out_of_range.
            // При крахе одного процесса остальные ранки зависнут на MPI_Startall!
            try {
                int j = local_dict.at(req.key_id); 
                m_peers[i].requested_columns.push_back(j);
            } catch (const std::out_of_range&) {
                std::cerr << "[Rank " << m_rank << "] Критическая ошибка: Ключ id=" 
                          << req.key_id << " не найден в локальном словаре!" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    m_loop_requests.clear();
    m_loop_requests.reserve(num_targets * 2);
    for (int i = 0; i < num_targets; ++i) {
        MPI_Recv_init(m_peers[i].recv_buffer.data(), m_peers[i].recv_buffer.size(), 
                      m_mpi_type, m_peers[i].rank, 50, MPI_COMM_WORLD, &m_peers[i].recv_msg);
        
        MPI_Send_init(m_peers[i].send_buffer.data(), m_peers[i].send_buffer.size(), 
                      m_mpi_type, m_peers[i].rank, 50, MPI_COMM_WORLD, &m_peers[i].send_msg);

        m_loop_requests.push_back(m_peers[i].recv_msg);
        m_loop_requests.push_back(m_peers[i].send_msg);
    }
}

    // --- ОБМЕН НА КАЖДОМ ШАГЕ ЦИКЛА ---
    void exchange_step(const Matrix<T>& matrix) {
        const T* flat_matrix_ptr = matrix.data();
        int cols = matrix.cols();
        int rows = matrix.rows();

        // Сборка столбцов в буфер отправки
        for (auto& peer : m_peers) {
            size_t dest_idx = 0;
            // Пробегаем по всем столбцам, которые у нас попросили
            for (int j : peer.requested_columns) {
                // Пробегаем по всем строкам нашей матрицы для этого столбца
                for (int r = 0; r < rows; ++r) {
                    size_t src_idx = static_cast<size_t>(r) * cols + j;
                    peer.send_buffer[dest_idx++] = flat_matrix_ptr[src_idx];
                }
            }
        }

        // Физическая отправка пакетов по сети
        MPI_Startall(m_loop_requests.size(), m_loop_requests.data());
        MPI_Waitall(m_loop_requests.size(), m_loop_requests.data(), MPI_STATUSES_IGNORE);
    }

    // Возвращает указатель на начало данных i-го столбца, полученного от peer_index
    // col_sub_index - это порядковый номер столбца среди тех, что вы запросили у ЭТОГО соседа (0, 1, 2...)
    const T* get_received_column(int peer_index, int col_sub_index, int& out_column_size) const {
        out_column_size = m_peers[peer_index].peer_matrix_rows;
        size_t offset = static_cast<size_t>(col_sub_index) * out_column_size;
        return &m_peers[peer_index].recv_buffer[offset];
    }
};
