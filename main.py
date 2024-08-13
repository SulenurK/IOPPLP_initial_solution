import numpy as np
from fields import ITEM_FIELDS
from order import Order, global_order_list
from batch import Batch, init_item_orientation
from sim_ann import calculate_route_makespan
import random
from typing import List
import logging
from config import parser
from util import read_coordinates_file, read_orders_file
import time
import pandas as pd

origin_distance = []
distance_matrix = []
batches = list[Batch]()  # batches = []

def batchToString(batch: Batch):
    s = f"Batch({batch.id}): 0, "
    for i, order_id in enumerate(batch.orders):
        r = batch.is_order_reversed[i]
        reverse_indicator = "'" if batch.is_order_reversed[i] else ""
        route_makespan = calculate_route_makespan(global_order_list[order_id].route, distance_matrix)
        s += f" o{order_id}({global_order_list[order_id].route}){reverse_indicator}[{route_makespan}] ,"
    s += " 0"
    s += f", Makespan: [{calculate_batch_makespan(batch.orders, batch.is_order_reversed)}]"
    return s


def calculate_batch_makespan(batch_orders, reverse=None):
    route = []
    if reverse is None:
        reverse = len(batch_orders) * [False]
    for i, order_id in enumerate(batch_orders):
        r = global_order_list[order_id].route
        if reverse[i]:
            r = r.copy()
            r.reverse()
        route.extend(r)
    return origin_distance[route[0]] + calculate_route_makespan(route, distance_matrix) + origin_distance[route[-1]]

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

def calculate_makespan(num_pickers: int, batch_list: List[Batch]):
    job_worker_pairs = [(f"Batch {b.id}", float(calculate_batch_makespan(b.orders, b.is_order_reversed))) for b in batch_list]

    # İş-işçi çiftlerini iş sürelerine göre azalan sırayla sıralama
    sorted_job_worker_pairs = sorted(job_worker_pairs, key=lambda x: x[1], reverse=True)

    # Makineler listesi (her biri başlangıçta 0 süreye sahip)
    pickers = [0] * num_pickers

    # Her iş için en az yük altındaki makineyi bulma ve iş-işçi çiftini o makineye atama
    job_allocation = [[] for _ in range(num_pickers)]

    for job_worker in sorted_job_worker_pairs:
        _, job_duration = job_worker
        # En az yük altındaki makineyi bul
        min_machine_index = pickers.index(min(pickers))
        # O makineye iş-işçi çiftini ata (süresini ekle)
        pickers[min_machine_index] += job_duration
        job_allocation[min_machine_index].append(job_worker)

    # Sonuçları yazdırma
    for i, machine_jobs in enumerate(job_allocation):
        machine_jobs_str = ', '.join([f"{worker} ({duration})" for worker, duration in machine_jobs])
        logging.debug(f"Makine {i+1}: {machine_jobs_str}, Toplam süre: {sum(duration for _, duration in machine_jobs)}")

    # Makespan'i yazdırma
    logging.debug(f"Toplam tamamlanma süresi (Makespan): {max(pickers)}")

    return job_allocation, max(pickers)

def assign_orders_to_batches(order_list, batch_list: List[Batch]):
    for order_id in order_list:
        assigned = False

        for b in batch_list:
            assigned = b.assign(global_order_list[order_id].items)
            if assigned:
                break

        if not assigned:
            b_new = Batch()
            b_new.assign(global_order_list[order_id].items)
            batch_list.append(b_new)

def run_foreach_batch(batch_list):
    logging.debug("=== Batches ===")
    for b in batch_list:
        logging.debug(batchToString(b))

    # Simulated annealing fonksiyonunun kaldırılmasıyla burada herhangi bir ek işlem yapmıyoruz.

    logging.debug("=== Batches After Processing ===")
    for b in batch_list:
        logging.debug(batchToString(b))

def main(args):
    global batches, distance_matrix, origin_distance

    start_time = time.time()

    np.random.seed(seed=args.seed)

    orders_items_list = read_orders_file(args.orders_file)
    num_orders = max([order_id[ITEM_FIELDS.ORDER_ID] for order_id in orders_items_list]) + 1
    init_item_orientation(len(orders_items_list))

    distance_matrix, origin_distance = read_coordinates_file(args.coordinates_file)

    # Calculate savings
    savings = []
    for i in range(len(origin_distance)):
        for j in range(i + 1, len(origin_distance)):
            saving = origin_distance[i] + origin_distance[j] - distance_matrix[i][j]
            savings.append((saving, i, j))

    savings.sort(reverse=True)

    # Initialize orders
    for order_id in range(num_orders):
        o = Order([i for i in orders_items_list if i[ITEM_FIELDS.ORDER_ID] == order_id], savings)
        o.route = o.route  # Mevcut rotayı kullan
        global_order_list.append(o)
        print(f"order{order_id}: {o.items}")

    current_order_list = list(range(0, len(global_order_list)))
    assign_orders_to_batches(current_order_list, batches)
    run_foreach_batch(batches)
    current_makespan = calculate_makespan(args.num_pickers, batches)

    print(f"Final makespan: {current_makespan[1]}")

    end_time = time.time()
    duration = end_time - start_time

    if args.out_file:
        out_file = args.out_file
    else:
        out_file = f"{args.orders_file}.output.csv"

    # Simulated annealing olmadığı için basit sonuçları kaydediyoruz.
    np.savetxt(
        out_file,
        np.array([[0, current_makespan[1]]]), 
        fmt="%d", 
        delimiter=",", 
        header="k, Makespan", 
        comments=""
    )

 

    args.start_time = start_time
    args.end_time = end_time
    args.duration = duration

    dictionary_to_df = pd.DataFrame(args.__dict__, index=[0])
    with open(f"{args.orders_file}.test_runs.txt", 'a') as f:
        dictionary_to_df.to_csv(f, mode="a", index=False, header=not f.tell())


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Args: {args}")
    main(args)
