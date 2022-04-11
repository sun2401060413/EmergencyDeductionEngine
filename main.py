from lib.interface import interface
from modules.simulation.simulator import simulator
if(__name__ == "__main__"):
    print("Launching the Emergency Deduction System...")
    simulator()
    interface("localhost", 55576)
