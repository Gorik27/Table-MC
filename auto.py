#!/home/user/miniconda3/bin/python
import concurrent.futures
import multiprocessing
import os
import shlex
import shutil
import glob
import signal
import subprocess
import sys
import numpy as np
import argparse

MAX_NUM_WORKERS = 50
parser = argparse.ArgumentParser()
parser.add_argument("pos_args", type=str, nargs="+", help='<chemical_specie_1> <mu_min_1> <mu_max_1> ...')
parser.add_argument("--num", type=int, nargs="+", default=[50], help='number of mu points for each specie')
parser.add_argument("-T", type=float, default=3, help='temperature (kT) in kJ/mol (300K = 2.5 kJ/mol)')
parser.add_argument("-m", "--matrix", type=str, default="Ag", help='chrmical specie of matrix')
parser.add_argument("-s", "--steps", type=int, default=500000000, help='number of MC steps')
parser.add_argument("-j", type=int, default=None, help='number of workers')
parser.add_argument("--print-each", type=int, default=None, help='number of MC steps between printing thermo to mc_output.txt. if is None print-each=steps/5000')
parser.add_argument("--dump-each", type=int, default=None, help='number of MC steps between savind dumps. if is None write only final dump')
parser.add_argument("--nrows", type=int, default=1, help='the number of rows (copies of each site type) in MC simulation')
parser.add_argument("--mu-precision", type=int, default=4, help='the number of decimal places for the mu value')
args = parser.parse_args()

steps = args.steps
if args.print_each is None:
    print_each = int(round(steps/5000))
else:
    print_each = args.print_each
kT = args.T
matrix_specie = args.matrix
nrows = args.nrows

mu_precision = args.mu_precision

if len(args.pos_args)%3 != 0:
    raise ValueError("Number of positional args must be the multiple of 3!")

n_solutes = len(args.pos_args)//3
elements = [args.pos_args[i*3] for i in range(n_solutes)]
mu_min = [float(args.pos_args[i*3+1]) for i in range(n_solutes)]
mu_max = [float(args.pos_args[i*3+2]) for i in range(n_solutes)]

if len(args.num)==n_solutes:
    MU_POINTS = args.num
elif len(args.num) == 1:
    MU_POINTS = [args.num[0]]*n_solutes
else:
    raise ValueError("number of mu points [--num] must be equal dimension to n_solutes or single value!")

if args.j is None:
    NUM_WORKERS = min(MAX_NUM_WORKERS, int(np.prod(MU_POINTS)))
else:
    NUM_WORKERS = args.j

# --- НАСТРОЙКИ ---
if args.dump_each is None:
    dump_each = steps # write only final dump
else:
    dump_each = args.dump_each
argument_string = f"mc_table -s {steps} --print-each {print_each} --dump-each {dump_each} -n {nrows} -T {kT} {matrix_specie} 0.0" 
COMMAND = shlex.split(argument_string)

def create_grid(*axes):
    if len(axes) == 1:
        return np.asarray(axes[0]).reshape(-1, 1)
    grid = np.meshgrid(*axes, indexing='ij')
    return np.stack(grid, axis=-1).reshape(-1, len(axes))
# Округляем значения, чтобы избежать багов с плавающей точкой
mu_axes = [np.round(np.linspace(mu_min[i], mu_max[i], num=MU_POINTS[i]), mu_precision) for i in range(n_solutes)]
MU_VALUES = create_grid(*mu_axes)

FOLDER_TO_COPY = "restart"  
FOLDER_TO_RENAME = "dump"  
FILE_TO_RENAME = "mc_output.txt"
INPUT_FILES = ["neighbors.txt", "es_*.txt", "eint_*.txt"]



RAM_DISK_ROOT = os.path.join("/dev/shm", f"mc_table_run_{os.getpid()}")

_is_leader_worker = False
_stopping = False
_current_process = None

def _sigterm_handler(signum, frame):
    global _stopping
    _stopping = True
    p = _current_process
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)   # numactl делает exec, pid совпадает с pid расчёта
        except ProcessLookupError:
            pass

def init_worker(counter, lock):
    global _is_leader_worker
    with lock:
        if counter.value == 0:
            counter.value = 1
            _is_leader_worker = True
        else:
            _is_leader_worker = False
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _sigterm_handler)

def run_single_calculation(task_args):
    global _current_process
    if _stopping:          # задачи, которые пул успел раздать после Ctrl+C
        return
    idx, vals = task_args
    if idx % 2 == 0:
        # Ограничиваем ядрами 0-25 и памятью сокета 0
        numa_prefix = ["numactl", "--cpunodebind=0", "--membind=0"]
    else:
        # Ограничиваем ядрами 26-51 и памятью сокета 1
        numa_prefix = ["numactl", "--cpunodebind=1", "--membind=1"]
        

    """Функция для расчета одного значения mu в ОЗУ с гарантированным бэкапом."""
    root_dir = os.getcwd()
    vals_string = "_".join([str(val) for val in vals])
    work_dir = os.path.join(RAM_DISK_ROOT, f"run_mu_{vals_string}")
    
    success_saved = False 
    process = None 

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Трехуровневый поиск рестарт файлов
    specific_restart_src = os.path.join(root_dir, f"{FOLDER_TO_COPY}_mu_{vals_string}")
    backup_restart_src = os.path.join(root_dir, f"BACKUP_mu_{vals_string}", FOLDER_TO_COPY)
    base_restart_src = os.path.join(root_dir, FOLDER_TO_COPY)
    dest_restart_folder = os.path.join(work_dir, FOLDER_TO_COPY)

    if os.path.exists(specific_restart_src):
        shutil.copytree(specific_restart_src, dest_restart_folder)
    elif os.path.exists(backup_restart_src):
        shutil.copytree(backup_restart_src, dest_restart_folder)
    elif os.path.exists(base_restart_src):
        shutil.copytree(base_restart_src, dest_restart_folder)

    # Копирование входных файлов
    for INPUT_FILE in INPUT_FILES:
        search_pattern = os.path.join(root_dir, INPUT_FILE)
        found_files = glob.glob(search_pattern)
        if found_files:
            for src_file_path in found_files:
                # Извлекаем точное имя файла (например, es_123.txt) из полного пути
                file_name = os.path.basename(src_file_path)
                shutil.copy2(src_file_path, os.path.join(work_dir, file_name))

    current_command = COMMAND.copy()

    for i in range(n_solutes):
        current_command.append(elements[i])
        current_command.append(str(vals[i]))

    current_command = numa_prefix + current_command

    try:
        if _is_leader_worker:
            print(f"\n[ВНИМАНИЕ] Процесс mu={vals_string} выбран ЛИДЕРОМ. Его вывод транслируется ниже:\n")
            print(current_command)
            process = subprocess.Popen(current_command, cwd=work_dir, stdout=None, stderr=None, start_new_session=True)
        else:
            process = subprocess.Popen(current_command, cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

        _current_process = process
        if _stopping:
            os.killpg(process.pid, signal.SIGTERM)
        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, current_command)

        # --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ (ШТАТНОЕ) ---
        work_folder_copy = os.path.join(work_dir, FOLDER_TO_COPY)
        if os.path.exists(work_folder_copy):
            copied_folder_name = os.path.join(root_dir, f"{FOLDER_TO_COPY}_mu_{vals_string}")
            if os.path.exists(copied_folder_name): shutil.rmtree(copied_folder_name)
            shutil.copytree(work_folder_copy, copied_folder_name)

        work_folder_rename = os.path.join(work_dir, FOLDER_TO_RENAME)
        if os.path.exists(work_folder_rename):
            renamed_folder_name = os.path.join(root_dir, f"{FOLDER_TO_RENAME}_mu_{vals_string}")
            if os.path.exists(renamed_folder_name): shutil.rmtree(renamed_folder_name)
            shutil.move(work_folder_rename, renamed_folder_name)

        work_file_rename = os.path.join(work_dir, FILE_TO_RENAME)
        if os.path.exists(work_file_rename):
            name, ext = os.path.splitext(FILE_TO_RENAME)
            renamed_file_name = os.path.join(root_dir, f"{name}_mu_{vals_string}{ext}")
            if os.path.exists(renamed_file_name): os.remove(renamed_file_name)
            shutil.move(work_file_rename, renamed_file_name)

        old_backup = os.path.join(root_dir, f"BACKUP_mu_{vals_string}")
        if os.path.exists(old_backup): shutil.rmtree(old_backup)

        success_saved = True 

    except Exception as e:
        print(f" Процесс mu {vals_string} остановлен.")
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except Exception:
                pass
        
    finally:
        _current_process = None
        # --- АВАРИЙНАЯ ЭВАКУАЦИЯ ДАННЫХ ---
        if not success_saved and os.path.exists(work_dir):
            backup_dir = os.path.join(root_dir, f"BACKUP_mu_{vals_string}")
            print(f"[БЭКАП] Спасение данных mu {vals_string} -> {backup_dir}")
            try:
                if os.path.exists(backup_dir): shutil.rmtree(backup_dir)
                shutil.copytree(work_dir, backup_dir)
            except Exception as be:
                print(f"Ошибка бэкапа для mu {vals_string}: {be}")

        if os.path.exists(work_dir):
            try: shutil.rmtree(work_dir)
            except Exception: pass


def main():
    print(f"Создание общей папки сессии в RAM: {RAM_DISK_ROOT}")
    os.makedirs(RAM_DISK_ROOT, exist_ok=True)

    print(f"Начало параллельных расчетов в RAM. Потоков: {NUM_WORKERS}")

    manager = multiprocessing.Manager()
    leader_counter = manager.Value('i', 0)
    leader_lock = manager.Lock()

    # Передаем функцию-инициализатор воркеров initializer=init_worker
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS, 
                                                initializer=init_worker,
                                                initargs=(leader_counter, leader_lock)) as executor:
        try:
            # Запускаем расчеты и оборачиваем в list(), чтобы вызов заблокировал главный поток
            indexed_args = list(enumerate(MU_VALUES))
            list(executor.map(run_single_calculation, indexed_args))
        except KeyboardInterrupt:
            print("\nПолучен Ctrl+C! Запускаем экстренное сохранение бэкапов и закрытие пула...")
            
            procs = list(executor._processes.values())
            executor.shutdown(wait=False, cancel_futures=True)  # отменяем очередь
            for p in procs:
                try: p.terminate()                               # SIGTERM воркерам
                except Exception: pass
            executor.shutdown(wait=True)
            if os.path.exists(RAM_DISK_ROOT):
                shutil.rmtree(RAM_DISK_ROOT, ignore_errors=True)
            print("\nДанные спасены в папки BACKUP_mu_*. Терминал свободен.")
            sys.exit(1)

    # Штатное удаление папки при успешном окончании
    if os.path.exists(RAM_DISK_ROOT):
        try: shutil.rmtree(RAM_DISK_ROOT)
        except Exception: pass
    print("\nВсе параллельные запуски успешно завершены!")


if __name__ == "__main__":
    main()
