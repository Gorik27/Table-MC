#include <iostream>
#include <string>
#include <iomanip>
#include <chrono>
#include <sys/ioctl.h> // Нужно для получения размера терминала в Linux/macOS
#include <unistd.h>    // Нужно для STDOUT_FILENO

class ProgressBar {
private:
    size_t total_steps;
    bool is_active;
    std::chrono::steady_clock::time_point start_time;

    // Функция, которая возвращает текущую ширину терминала в символах
    int get_terminal_width() const {
        struct winsize w;
        // Запрашиваем у ОС параметры окна для стандартного вывода
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) {
            return w.ws_col;
        }
        return 80; // Если размер определить не удалось (например, вывод перенаправлен), возвращаем дефолт
    }

    std::string format_duration(long long total_seconds) const {
        long long hours = total_seconds / 3600;
        long long minutes = (total_seconds % 3600) / 60;
        long long seconds = total_seconds % 60;

        std::string result = "";
        if (hours > 0) result += std::to_string(hours) + ":";
        if (minutes < 10 && hours > 0) result += "0";
        result += std::to_string(minutes) + ":";
        if (seconds < 10) result += "0";
        result += std::to_string(seconds);
        return result;
    }

public:
    // Теперь третий параметр (ширина) больше не нужен в конструкторе!
    ProgressBar(size_t total, bool active = true) 
        : total_steps(total), is_active(active) {
        if (is_active) {
            start_time = std::chrono::steady_clock::now();
        }
    }

    void update(size_t current_step) {
        if (!is_active || current_step == 0) return;

        auto now = std::chrono::steady_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        float progress = static_cast<float>(current_step) / total_steps;

        long long eta_seconds = 0;
        if (progress > 0.0f) {
            eta_seconds = static_cast<long long>((elapsed_seconds / progress) - elapsed_seconds);
        }

        // 1. Формируем правую (текстовую) часть, чтобы узнать её точную длину
        std::string stats = " " + std::to_string(static_cast<int>(progress * 100.0)) + "% "
                          + "(" + std::to_string(current_step) + "/" + std::to_string(total_steps) + ") "
                          + "[" + format_duration(elapsed_seconds) + "<" + format_duration(eta_seconds) + "]";

        // 2. Считаем доступное место под шкалу [████   ]
        int term_width = get_terminal_width();
        
        // Из всей ширины вычитаем длину текста, 2 символа под скобки [] и 1 под символ '\r'
        int bar_width = term_width - stats.length() - 3; 
        
        // Защита: если окно терминала слишком узкое, делаем шкалу хотя бы минимальной
        if (bar_width < 10) bar_width = 10; 

        // 3. Вычисляем сколько символов закрасить
        int pos = static_cast<int>(bar_width * progress);

        // 4. Отрисовка
        std::cout << "\r["; 
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "█";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "]" << stats << std::flush;
    }

    void finish() {
        if (is_active) {
            std::cout << std::endl;
        }
    }
};
