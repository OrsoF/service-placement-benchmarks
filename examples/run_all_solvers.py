from service_placement.benchmarks.smoke import run_all_solvers

if __name__ == "__main__":
    for row in run_all_solvers():
        print(row)
