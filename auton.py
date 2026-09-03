import concurrent.futures
import os
import shlex
import shutil
import signal
import subprocess
import sys
import numpy as np

# --- НАСТРОЙКИ ---
argument_string = "mpirun -np 1 --bind-to none /media/user/HDD/Projects/MC_table/mc_table -s 500000000 -n 1 -T 10 --mu 0"
MPI_COMMAND = shlex.split(argument_string)

# Округляем значения, чтобы избежать багов с плавающей точкой
ARG_VALUES = np.round(np.linspace(80, -20, num=50), 4)

FOLDER_TO_COPY = "restart"  
FOLDER_TO_RENAME = "dump"  
FILE_TO_RENAME = "mc_output.txt"

NUM_WORKERS = 50

RAM_DISK_ROOT = os.path.join("/dev/shm", f"mc_table_run_{os.getpid()}")


def init_worker():
    """Заставляет дочерние процессы игнорировать Ctrl+C.
    
    Благодаря этому они не умирают мгновенно, а корректно уходят в блок 
    except/finally для спасения бэкапов, когда главный процесс закрывает пул.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_single_calculation(task_args):
    idx, val = task_args
    if idx % 2 == 0:
        # Ограничиваем ядрами 0-25 и памятью сокета 0
        numa_prefix = ["numactl", "--cpunodebind=0", "--membind=0"]
    else:
        # Ограничиваем ядрами 26-51 и памятью сокета 1
        numa_prefix = ["numactl", "--cpunodebind=1", "--membind=1"]
        

    """Функция для расчета одного значения mu в ОЗУ с гарантированным бэкапом."""
    root_dir = os.getcwd()
    work_dir = os.path.join(RAM_DISK_ROOT, f"run_mu_{val}")
    
    success_saved = False 
    process = None 

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Трехуровневый поиск рестарт файлов
    specific_restart_src = os.path.join(root_dir, f"{FOLDER_TO_COPY}_mu_{val}")
    backup_restart_src = os.path.join(root_dir, f"BACKUP_mu_{val}", FOLDER_TO_COPY)
    base_restart_src = os.path.join(root_dir, FOLDER_TO_COPY)
    dest_restart_folder = os.path.join(work_dir, FOLDER_TO_COPY)

    if os.path.exists(specific_restart_src):
        shutil.copytree(specific_restart_src, dest_restart_folder)
    elif os.path.exists(backup_restart_src):
        shutil.copytree(backup_restart_src, dest_restart_folder)
    elif os.path.exists(base_restart_src):
        shutil.copytree(base_restart_src, dest_restart_folder)

    # Копирование входных файлов
    INPUT_FILES = ["new_neighbors.txt", "new_es.txt", "new_eint.txt"]
    for INPUT_FILE in INPUT_FILES:
        src_file_path = os.path.join(root_dir, INPUT_FILE)
        if os.path.exists(src_file_path):
            shutil.copy2(src_file_path, os.path.join(work_dir, INPUT_FILE))

    current_command = MPI_COMMAND.copy()
    for i, part in enumerate(current_command):
        if "mc_table" in part:
            current_command[i] = os.path.abspath(os.path.join(root_dir, part))

    current_command.append(str(val))
    # Склеиваем префиксnumactl и вашу основную команду
    current_command = numa_prefix + current_command

    try:
        is_leader = (val == ARG_VALUES[0])

        if is_leader:
            print(f"\n📡 [ВНИМАНИЕ] Процесс mu={val} выбран ЛИДЕРОМ. Его вывод транслируется ниже:\n")
            process = subprocess.Popen(current_command, cwd=work_dir, stdout=None, stderr=None)
        else:
            process = subprocess.Popen(current_command, cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, current_command)

        # --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ (ШТАТНОЕ) ---
        work_folder_copy = os.path.join(work_dir, FOLDER_TO_COPY)
        if os.path.exists(work_folder_copy):
            copied_folder_name = os.path.join(root_dir, f"{FOLDER_TO_COPY}_mu_{val}")
            if os.path.exists(copied_folder_name): shutil.rmtree(copied_folder_name)
            shutil.copytree(work_folder_copy, copied_folder_name)

        work_folder_rename = os.path.join(work_dir, FOLDER_TO_RENAME)
        if os.path.exists(work_folder_rename):
            renamed_folder_name = os.path.join(root_dir, f"{FOLDER_TO_RENAME}_mu_{val}")
            if os.path.exists(renamed_folder_name): shutil.rmtree(renamed_folder_name)
            shutil.move(work_folder_rename, renamed_folder_name)

        work_file_rename = os.path.join(work_dir, FILE_TO_RENAME)
        if os.path.exists(work_file_rename):
            name, ext = os.path.splitext(FILE_TO_RENAME)
            renamed_file_name = os.path.join(root_dir, f"{name}_mu_{val}{ext}")
            if os.path.exists(renamed_file_name): os.remove(renamed_file_name)
            shutil.move(work_file_rename, renamed_file_name)

        old_backup = os.path.join(root_dir, f"BACKUP_mu_{val}")
        if os.path.exists(old_backup): shutil.rmtree(old_backup)

        success_saved = True 

    except Exception as e:
        print(f" Процесс mu {val} остановлен.")
        if process and process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=1)
            except Exception:
                pass
        
    finally:
        # --- АВАРИЙНАЯ ЭВАКУАЦИЯ ДАННЫХ ---
        if not success_saved and os.path.exists(work_dir):
            backup_dir = os.path.join(root_dir, f"BACKUP_mu_{val}")
            print(f"⚠️  [БЭКАП] Спасение данных mu {val} -> {backup_dir}")
            try:
                if os.path.exists(backup_dir): shutil.rmtree(backup_dir)
                shutil.copytree(work_dir, backup_dir)
            except Exception as be:
                print(f"Ошибка бэкапа для mu {val}: {be}")

        if os.path.exists(work_dir):
            try: shutil.rmtree(work_dir)
            except Exception: pass


def main():
    print(f"Создание общей папки сессии в RAM: {RAM_DISK_ROOT}")
    os.makedirs(RAM_DISK_ROOT, exist_ok=True)

    print(f"Начало параллельных расчетов в RAM. Потоков: {NUM_WORKERS}")

    # Передаем функцию-инициализатор воркеров initializer=init_worker
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        try:
            # Запускаем расчеты и оборачиваем в list(), чтобы вызов заблокировал главный поток
            indexed_args = list(enumerate(ARG_VALUES))
            list(executor.map(run_single_calculation, indexed_args))
        except KeyboardInterrupt:
            print("\n🛑 Получен Ctrl+C! Запускаем экстренное сохранение бэкапов и закрытие пула...")
            
            # Быстро останавливаем пул, заставляя воркеры прервать выполнение и уйти в блоки except/finally
            executor.shutdown(wait=True, cancel_futures=True)

            if os.path.exists(RAM_DISK_ROOT):
                try: shutil.rmtree(RAM_DISK_ROOT)
                except Exception: pass
            
            print("\nДанные спасены в папки BACKUP_mu_*. Терминал свободен.")
            sys.exit(1)

    # Штатное удаление папки при успешном окончании
    if os.path.exists(RAM_DISK_ROOT):
        try: shutil.rmtree(RAM_DISK_ROOT)
        except Exception: pass
    print("\nВсе параллельные запуски успешно завершены!")


if __name__ == "__main__":
    main()
