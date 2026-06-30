from model.precision import set_precision

# Set the decimal precision used by the simulation.
set_precision(200)

from simulations import *

if __name__ == "__main__":
    try:
        number_of_simulation = int(input("Enter the simulation number: "))
        func_name = f"simulation{number_of_simulation}"
        if func_name in globals():
            globals()[func_name]()
        else:
            print(f"Simulation {number_of_simulation} is not defined.")

    except Exception as e:
        print(f"An error occurred: {e}")
