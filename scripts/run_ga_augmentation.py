import argparse
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd
import pygad
import yaml

from plant_stress_aug.train_eval import fit_one_cycle


try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


last_fitness = 0
CONFIG = None
AUGMENTATION_NAMES = None
OUTPUT_DIR = None


def fitness_func(ga_instance, chromosome, chromosome_idx):
    _, mpca = fit_one_cycle(
        chromosome=chromosome,
        chromosome_idx=chromosome_idx,
        config=CONFIG,
        augmentation_names=AUGMENTATION_NAMES,
        output_dir=OUTPUT_DIR,
    )
    return mpca


def on_generation(ga_instance):
    global last_fitness
    best_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {best_fitness}")
    print(f"Change     = {best_fitness - last_fitness}")
    last_fitness = best_fitness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    global CONFIG, AUGMENTATION_NAMES, OUTPUT_DIR
    with open(args.config, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)

    AUGMENTATION_NAMES = CONFIG["augmentation"]["transforms"]
    OUTPUT_DIR = args.output_dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    num_classes = CONFIG["model"]["num_classes"]
    num_genes = num_classes * len(AUGMENTATION_NAMES)

    ga_instance = pygad.GA(
        sol_per_pop=CONFIG["ga"]["sol_per_pop"],
        num_genes=num_genes,
        gene_type=[float, 1],
        num_generations=CONFIG["ga"]["num_generations"],
        num_parents_mating=CONFIG["ga"]["num_parents_mating"],
        fitness_func=fitness_func,
        init_range_low=CONFIG["ga"]["init_range_low"],
        init_range_high=CONFIG["ga"]["init_range_high"],
        parent_selection_type=CONFIG["ga"]["parent_selection_type"],
        keep_elitism=CONFIG["ga"]["keep_elitism"],
        crossover_type=CONFIG["ga"]["crossover_type"],
        mutation_type=CONFIG["ga"]["mutation_type"],
        mutation_by_replacement=True,
        mutation_percent_genes=CONFIG["ga"]["mutation_percent_genes"],
        random_mutation_min_val=CONFIG["ga"]["random_mutation_min_val"],
        random_mutation_max_val=CONFIG["ga"]["random_mutation_max_val"],
        on_generation=on_generation,
        save_best_solutions=True,
        save_solutions=True,
        parallel_processing=["process", CONFIG["ga"]["parallel_workers"]],
    )

    t1 = time.time()
    ga_instance.run()
    t2 = time.time()
    print(f"Time is {t2 - t1}")

    pd.DataFrame(ga_instance.best_solutions).to_csv(Path(OUTPUT_DIR) / "best_solutions.csv", index=False)
    pd.DataFrame(ga_instance.solutions).to_csv(Path(OUTPUT_DIR) / "solutions.csv", index=False)
    pd.DataFrame(ga_instance.best_solutions_fitness).to_csv(Path(OUTPUT_DIR) / "best_solutions_fitness.csv", index=False)
    ga_instance.plot_fitness(save_dir=str(Path(OUTPUT_DIR) / "fitness.png"))
    ga_instance.save(filename=str(Path(OUTPUT_DIR) / "ga_instance"))

    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print("Solution", solution)
    print(f"Fitness value of the best solution = {solution_fitness}")


if __name__ == "__main__":
    main()
