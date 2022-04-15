from lib.interface import interface
from modules.simulation.simulator import Simulator
if(__name__ == "__main__"):
    print("Launching the Emergency Deduction System...")
    Simulator()
    interface("localhost", 55576)
