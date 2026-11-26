import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_dag import init_dag_phase

def main():
    init_dag_phase()
    print("Executing phase HPO...")
    # placeholder

if __name__ == "__main__":
    main()
