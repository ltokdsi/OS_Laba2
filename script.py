import sys
import time
import random
import multiprocessing as mp

def generator_worker(n, q):
    """
    Процесс-генератор.
    Генерирует нижнетреугольную матрицу для неравномерной нагрузки.
    """
    matrix = []
    for row in range(n):
        for col in range(n):
            if col <= row:
                matrix.append(random.randint(1, 9))
            else:
                matrix.append(0)
    q.put(matrix)

def compute_worker(worker_id, n, p, T, mode, base_A, buf1, buf2,
                   progress_counter, dyn_counters, barrier):
    """
    Рабочий процесс-вычислитель.
    """
    MOD = 1000000007

    for step in range(1, p):
        if step % 2 != 0:
            read_buf = buf1
            write_buf = buf2
        else:
            read_buf = buf2
            write_buf = buf1

        if mode == 'static':
            rows_per_worker = n // T
            remainder = n % T

            start_row = worker_id * rows_per_worker + min(worker_id, remainder)
            end_row = start_row + rows_per_worker + (1 if worker_id < remainder else 0)

            for i in range(start_row, end_row):
                for j in range(n):
                    s = 0
                    # Идем только до i+1, так как матрица нижнетреугольная
                    for k in range(i + 1):
                        s += read_buf[i * n + k] * base_A[k * n + j]
                        s %= MOD
                    write_buf[i * n + j] = s

                with progress_counter.get_lock():
                    progress_counter.value += 1

        elif mode == 'dynamic':
            while True:
                with dyn_counters.get_lock():
                    i = dyn_counters[step]
                    dyn_counters[step] += 1

                if i >= n:
                    break

                for j in range(n):
                    s = 0
                    # Идем только до i+1, так как матрица нижнетреугольная
                    for k in range(i + 1):
                        s += read_buf[i * n + k] * base_A[k * n + j]
                        s %= MOD
                    write_buf[i * n + j] = s

                with progress_counter.get_lock():
                    progress_counter.value += 1

        barrier.wait()

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Использование: python script.py <размер_n> <степень_p> <кол_во_процессов> <static/dynamic>")
        sys.exit(1)

    n = int(sys.argv[1])
    p = int(sys.argv[2])
    T = int(sys.argv[3])
    mode = sys.argv[4]

    if p < 2:
        print("Для возведения в степень p должно быть >= 2")
        sys.exit(1)

    q = mp.Queue()
    gen_p = mp.Process(target=generator_worker, args=(n, q))

    ipc_start = time.perf_counter()
    gen_p.start()
    matrix_A = q.get()
    gen_p.join()
    ipc_end = time.perf_counter()

    if T == 1 and mode == 'static': # Выводим время IPC только один раз для красоты
        print(f"Время передачи IPC: {ipc_end - ipc_start:.4f} сек")

    base_A = mp.Array('q', matrix_A, lock=False)
    buf1 = mp.Array('q', matrix_A, lock=False)
    buf2 = mp.Array('q', n * n, lock=False)

    progress_counter = mp.Value('i', 0)
    dyn_counters = mp.Array('i', p, lock=True)

    barrier = mp.Barrier(T)

    total_rows = (p - 1) * n
    workers = []

    start_time = time.perf_counter()

    for i in range(T):
        w = mp.Process(target=compute_worker,
                       args=(i, n, p, T, mode, base_A, buf1, buf2,
                             progress_counter, dyn_counters, barrier))
        workers.append(w)
        w.start()

    while True:
        with progress_counter.get_lock():
            current = progress_counter.value

        if current >= total_rows:
            break
        time.sleep(0.1)

    for w in workers:
        w.join()

    end_time = time.perf_counter()

    final_buf = buf2 if (p - 1) % 2 != 0 else buf1
    checksum = sum(final_buf) % 1000000007

    print(f"Время умножения: {end_time - start_time:.4f} сек")
    print(f"Контрольная сумма: {checksum}")